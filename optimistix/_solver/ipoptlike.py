from collections.abc import Callable
from typing import Any, Generic, Union

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, Int, PyTree, Scalar, ScalarLike

from .._custom_types import (
    Aux,
    Constraint,
    DescentState,
    EqualityOut,
    Fn,
    InequalityOut,
    SearchState,
    Y,
)
from .._minimise import AbstractMinimiser
from .._misc import (
    evaluate_constraint,
    feasible_step_length,
    filter_cond,
    lin_to_grad,
    max_norm,
    tree_clip,
    tree_full_like,
    tree_where,
    verbose_print,
)
from .._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .barrier import LogarithmicBarrier
from .bfgs import BFGS_B, identity_pytree
from .boundary_maps import ClosestFeasiblePoint
from .filtered import IPOPTLikeFilteredLineSearch


# Some global flags strictly for use during development, these will be removed later.
# I'm introducing them here to be able to selectively enable certain special features,
# for use in testing and debugging.
# Note that it generally does not make sense to enable the filtered line search without
# enabling the feasibility restoration too, since this is currently the only place that
# the filter gets re-set. (We don't support the heuristic filter reset in the search
# yet.)
SECOND_ORDER_CORRECTION = False
FEASIBILITY_RESTORATION = True
FILTERED_LINE_SEARCH = True


def _interior_tree_clip(
    tree: PyTree[ArrayLike],
    lower: PyTree[ArrayLike],
    upper: PyTree[ArrayLike],
    bound_push: ScalarLike,
    bound_frac: ScalarLike,
):
    """Clip the tree to the strict interior of the feasible region, accounting for the
    value of the bounds and the distance between the bounds.

    The PyTree structures of `tree`, `lower`, and `upper` must match. Infinite bounds
    are acceptable. Note that this function assumes that upper >= lower for all elements
    but that is not checked here.

    The offset is determined as follows:

        offset = min( max( 1, abs(bound value) ), bound_frac * width )

    where width is the distance between the bounds, defined as (upper - lower), and we
    scale the offset by the value of the bounds if that value is larger than 1.0. (This
    is to ensure that we do not inadvertently clip to a zero bound, in which case the
    barrier function returns an infinite value.)

    `bound_push` and `bound_frac` correspond to the parameters `kappa_1` and `kappa_2`
    in the IPOPT implementation paper (doi: 10.1007/s10107-004-0559-y). Here we use the
    names currently in use in the IPOPT code.
    """
    # TODO: add runtime error if upper < lower, since the strict interior is not defined
    # in that case.

    def offset(x, width):
        scale = jnp.asarray(bound_push) * jnp.where(jnp.abs(x) < 1, 1, jnp.abs(x))
        safe_scale = jnp.where(jnp.isfinite(scale), scale, 1.0)

        safe_width = jnp.where(jnp.isfinite(width), width, 1.0)
        width_factor = jnp.asarray(bound_frac) * safe_width

        offset = jnp.where(width_factor < safe_scale, width_factor, safe_scale)
        return offset

    width = (upper**ω - lower**ω).ω
    lower_offset = jtu.tree_map(offset, lower, width)
    upper_offset = jtu.tree_map(offset, upper, width)

    moved_lower = (tree**ω + lower_offset**ω).ω
    moved_upper = (tree**ω - upper_offset**ω).ω

    return tree_clip(tree, moved_lower, moved_upper)


def _make_kkt_operator(hessian, jacobian, input_structure):
    def kkt(inputs):
        y, duals = inputs
        y_step = (hessian.mv(y) ** ω + jacobian.T.mv(duals) ** ω).ω
        dual_step = jacobian.mv(y)
        return y_step, dual_step

    return lx.FunctionLinearOperator(kkt, input_structure)


def _bound_multiplier_steps(
    y_step: Y,
    barrier_gradients: tuple[Y, Y],
    barrier_hessians: tuple[lx.DiagonalLinearOperator, lx.DiagonalLinearOperator],
    bound_multipliers: tuple[Y, Y],
    bounds: tuple[Y, Y],
    offset: ScalarLike,
):
    """Reconstruct the steps in the bound multipliers from the step in the primal
    variable `y`. The computed steps are then truncated to the maximum feasible step
    length, defined as the step length at which the bound multipliers remain positive.

    Following the IPOPT implementation, the boundary multipliers are not further
    constrained, e.g. by truncating the step to the maximum feasible step length of the
    primal variable `y`.
    """
    lower, upper = bounds
    lower_grad, upper_grad = barrier_gradients
    lower_hess, upper_hess = barrier_hessians
    lower_mult, upper_mult = bound_multipliers

    def multiplier_update(multiplier, grad, transformed_step, bound):
        update = (multiplier**ω - grad**ω - transformed_step**ω).ω
        return tree_where(jtu.tree_map(jnp.isfinite, bound), update, 0.0)

    lower_step = multiplier_update(lower_mult, lower_grad, lower_hess.mv(y_step), lower)
    upper_step = multiplier_update(upper_mult, upper_grad, upper_hess.mv(y_step), upper)

    _zero = tree_full_like(lower, 0.0)
    lower_step_size = feasible_step_length(lower_mult, _zero, lower_step, offset=offset)
    upper_step_size = feasible_step_length(upper_mult, _zero, upper_step, offset=offset)

    max_step_size = jnp.min(jnp.array([lower_step_size, upper_step_size]))
    lower_step = (max_step_size * lower_step**ω).ω
    upper_step = (max_step_size * upper_step**ω).ω
    return lower_step, upper_step


class _IPOPTLikeDescentState(
    eqx.Module, Generic[Y, EqualityOut, InequalityOut], strict=True
):
    step: PyTree
    result: RESULTS


# TODO: Make the documentation a bit more comprehensive here!
# And also make it a little cleaner :D
class IPOPTLikeDescent(
    AbstractDescent[Y, FunctionInfo.EvalGradHessian, _IPOPTLikeDescentState],
    strict=True,
):
    """A descent method, closely inspired by IPOPT. In this descent, we solve a primal-
    dual Karush-Kuhn-Tucker (KKT) system to compute steps in the primal variable `y` and
    the dual variables for the constraints and bounds. The linear system we are solving
    is condensed to express the steps in the bound multipliers as a function of the step
    in the primal optimisation variable `y`. This allows us to solve a smaller symmetric
    system.

    This descent currently requires bounds on all elements of `y`. Infinite bounds may
    be used.

    The integration of inequality constraints follows the approach outlined in Chapter
    19.3 of Nocedal & Wright, Numerical Optimisation, 2nd edition, page 569, but flips
    the sign convention in the Lagrangian from subtraction of the constraint terms to
    addition to match the IPOPT convention for equality constraints.

    ??? Differences to IPOPT implementation

    This descent is a simplified implementation of the step computation in IPOPT. The
    modifications are as follows:

    - The steps in the bound multipliers are constrained by the step size in the primal
        variable `y`, only the computation of the maximum feasible step length is
        uncoupled from the steps taken in `y`. This would be a simple change, but for
        now it is not clear what the advantages would be.

    ??? cite "References"

        Updating the bound multipliers from the solution of the condensed system is
        described in:

        ```bibtex
        @article{wachter2006implementation,
        author = {Wächter, Andreas and Biegler, Lorenz T.},
            title = {On the implementation of a primal-dual interior point filter line
                     search algorithm for large-scale nonlinear programming},
            journal = {Mathematical Programming},
            volume = {106},
            number = {1},
            pages = {25-57},
            year = {2006},
            doi = {10.1007/s10107-004-0559-y},
        }
        ```
    """

    linear_solver: lx.AbstractLinearSolver = lx.SVD()
    minimum_offset: ScalarLike = 0.99  # tau_min in IPOPT

    def init(  # pyright: ignore (iterate)
        self, iterate, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _IPOPTLikeDescentState:
        if f_info_struct.bounds is None:
            raise ValueError(
                "IPOPTLikeDescent requires bounds on the optimisation variable `y`. "
                "To use this descent without bound constraints, pass infinite bounds "
                "on all elements of `y`."
            )
        if f_info_struct.constraint_residual is None:
            raise ValueError(
                "IPOPTLikeDescent requires constraints on the optimisation variable "
                "`y`. This descent can currently not be used without any constraints, "
                "but it can be used with either inequality or equality constraints."
            )
        else:
            return _IPOPTLikeDescentState(iterate, RESULTS.successful)

    def query(  # pyright: ignore  (iterate)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _IPOPTLikeDescentState,
    ) -> _IPOPTLikeDescentState:
        assert f_info.bounds is not None
        assert f_info.constraint_residual is not None
        assert f_info.constraint_jacobians is not None

        y, constraint_multipliers, bound_multipliers, barrier_parameter = iterate

        # Compute the barrier gradients and Hessians
        # Note Johanna: we're currently computing the gradient of the barrier term twice
        # (the other time is in the filtered line search). The computation is cheap
        # (a subtraction, a division, and a multiplication), but I don't love that we
        # do this in different places. The price to pay to do it all in solver.step,
        # however, would be to either pass a much more complex f_info to the descent, or
        # to lose the flexibility of constructing a full, rather than a condensed KKT
        # system. So I think on balance I'd rather be doing these things twice.
        # (Ideas very welcome!)
        barrier = LogarithmicBarrier(f_info.bounds)
        barrier_gradients = barrier.grads(y, barrier_parameter)
        lower_barrier_grad, upper_barrier_grad = barrier_gradients
        barrier_hessians = barrier.primal_dual_hessians(y, bound_multipliers)
        lower_barrier_hessian, upper_barrier_hessian = barrier_hessians

        grad = (f_info.grad**ω + lower_barrier_grad**ω + upper_barrier_grad**ω).ω
        hessian = f_info.hessian + lower_barrier_hessian + upper_barrier_hessian

        # Construct and solve the condensed KKT system
        (equality_dual, inequality_dual) = constraint_multipliers
        equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
        equality_residual, inequality_residual = f_info.constraint_residual
        input_structure = jax.eval_shape(lambda: (y, equality_residual))
        kkt_operator = _make_kkt_operator(hessian, equality_jacobian, input_structure)

        dual_term = equality_jacobian.T.mv(equality_dual)  # pyright: ignore
        vector = (
            (-(grad**ω) - dual_term**ω).ω,
            (-(equality_residual**ω)).ω,
        )
        out = lx.linear_solve(kkt_operator, vector, self.linear_solver)

        y_step, equality_multiplier_step = out.value
        result = RESULTS.promote(out.result)

        # Postprocess the result: truncate to feasible step length
        offset = jnp.min(jnp.array([self.minimum_offset, 1 - barrier_parameter]))
        lower, upper = f_info.bounds
        lower_max_step_size = feasible_step_length(y, lower, y_step, offset=offset)
        upper_max_step_size = feasible_step_length(y, upper, y_step, offset=offset)

        max_step_size = jnp.min(jnp.array([lower_max_step_size, upper_max_step_size]))

        y_step = (max_step_size * y_step**ω).ω
        equality_multiplier_step = (max_step_size * equality_multiplier_step**ω).ω
        dual_steps = (
            equality_multiplier_step,
            tree_full_like(inequality_residual, 0.0),
        )

        # TODO: barrier parameter is an iterate that does not get updated in the descent

        # Finally, compute the steps in the bound multipliers from the primal step
        barrier_terms = (barrier_gradients, barrier_hessians, bound_multipliers)
        b_steps = _bound_multiplier_steps(y_step, *barrier_terms, f_info.bounds, offset)
        keep_barrier_parameter = tree_full_like(barrier_parameter, 0.0)

        iterate_step = (y_step, dual_steps, b_steps, keep_barrier_parameter)

        return _IPOPTLikeDescentState(
            iterate_step,
            result,
        )

    def step(
        self, step_size: Scalar, state: _IPOPTLikeDescentState
    ) -> tuple[Y, RESULTS]:
        # Note: by scaling the complete step with the step size, we do not presently
        # allow the bound multipliers to remain at their maximum feasible step length,
        # which is supported in IPOPT. Adding this feature would require unwrapping the
        # iterate, scaling the primal step and the steps in the constraint multipliers,
        # and then returning the half-updated step. This is easily done, but for now it
        # is not clear how much it helps - therefore we leave it be for now.
        return (step_size * state.step**ω).ω, state.result


# TODO: I don't support the automatic scaling of the norms yet (and am not sure if I
# should). In IPOPT, this is factor is at least 100, per the documentation and the paper

# TODO: I think we should raise errata when y0 is infeasible with respect to the
# inequality constraints, since this would mean that we have negative slack values.
# It would be good to also provide an easy-to-use utility function that users can use to
# compute an initial point that is feasible with respect to the inequality constraints.
# Right now we don't raise an error for this and I don't think this is acceptable, since
# we do not in fact know for sure that we do the right thing in this case.


# TODO: I think it would make sense to have an `AbstractTermination` that just takes
# an iterate, and a function info, and has a norm.
def _error(
    iterate: PyTree,  # pyright: ignore   # TODO API for termination criteria
    f_info: FunctionInfo.EvalGradHessian,
    barrier_parameter,  # TODO: no longer needed if we have a generalised Y.
    tol,
    norm,
):
    assert f_info.constraint_residual is not None
    assert f_info.constraint_jacobians is not None

    y, (equality_dual, inequality_dual), (lb_dual, ub_dual), barrier = iterate

    equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
    dual_term = equality_jacobian.T.mv(equality_dual)  # pyright: ignore
    if inequality_jacobian is not None:
        dual_term = (dual_term**ω + inequality_jacobian.T.mv(inequality_dual) ** ω).ω
    optimality_error = norm(
        jtu.tree_map(
            lambda a, b, c, d: a + b - c + d, f_info.grad, dual_term, lb_dual, ub_dual
        )
    )

    # TODO: implement support for inequality residuals
    equality_residual, inequality_residual = f_info.constraint_residual
    constraint_norm = norm(equality_residual)
    if inequality_residual is not None:
        inequality_violation = tree_where(
            # Only count violations if the residual is less than zero
            # Alternatively transform with slack! # TODO
            jtu.tree_map(lambda x: jnp.where(x < 0, x, 0.0), inequality_residual),
            inequality_residual,
            0.0,
        )
        constraint_norm += norm(inequality_violation)

    ### TO BE REFACTORED (made a separate optx.one_norm) -------------------------------
    def _one_norm(x):
        absolute_values = jnp.where(jnp.isfinite(x), jnp.abs(x), 0.0)
        return jnp.sum(absolute_values)

    multiplier_norm = jtu.tree_map(_one_norm, (equality_dual, inequality_dual))
    multiplier_norm, _ = jfu.ravel_pytree(multiplier_norm)
    multiplier_norm = jnp.sum(multiplier_norm)

    num_multipliers = jtu.tree_map(jnp.isfinite, (equality_dual, inequality_dual))
    num_multipliers, _ = jfu.ravel_pytree(num_multipliers)
    num_multipliers = jnp.sum(num_multipliers)

    bound_multiplier_norm = jtu.tree_map(_one_norm, (lb_dual, ub_dual))
    bound_multiplier_norm, _ = jfu.ravel_pytree(bound_multiplier_norm)
    bound_multiplier_norm = jnp.sum(bound_multiplier_norm)

    num_bounds = jtu.tree_map(jnp.isfinite, (lb_dual, ub_dual))  # Or check bounds
    num_bounds, _ = jfu.ravel_pytree(num_bounds)
    num_bounds = jnp.sum(num_bounds)
    # ----------------------------------------------------------------------------------

    summed_norms = multiplier_norm + bound_multiplier_norm
    summed_scaled_norm = summed_norms / (num_multipliers + num_bounds)
    scaling = jnp.where(summed_scaled_norm > 100.0, summed_scaled_norm, 100.0) / 100.0

    optimality_error = optimality_error / scaling

    # Aggregate errata - probably move this for better flow
    errata = (optimality_error, constraint_norm)

    lower, upper = f_info.bounds  # pyright: ignore (bounds not None)  # TODO
    lower_diff = (y**ω - lower**ω).ω
    upper_diff = (upper**ω - y**ω).ω
    lower_error = (lower_diff**ω * lb_dual**ω - barrier).ω
    upper_error = (upper_diff**ω * ub_dual**ω - barrier).ω
    finite_lower = jtu.tree_map(jnp.isfinite, lower)
    finite_upper = jtu.tree_map(jnp.isfinite, upper)
    lower_error = tree_where(finite_lower, lower_error, 0.0)
    upper_error = tree_where(finite_upper, upper_error, 0.0)

    # TODO: scale the lower error

    # TODO question: lower and upper error separat or joint?
    errata += (norm(lower_error), norm(upper_error))

    return jnp.max(jnp.asarray(errata))


class Iterate(eqx.Module, Generic[Y, EqualityOut, InequalityOut], strict=True):
    y_eval: Y  # TODO: or more concisely rename y?
    slack: InequalityOut
    multipliers: tuple[EqualityOut, InequalityOut]
    bound_multipliers: tuple[Y, Y]
    barrier: ScalarLike


class _IPOPTLikeState(
    eqx.Module, Generic[Y, Aux, SearchState, DescentState], strict=True
):
    # Updated every search step
    first_step: Bool[Array, ""]
    iterate: PyTree  # TODO: figuring out how to optimise over pairs (primal, dual)
    search_state: SearchState
    # Updated after each descent step
    f_info: FunctionInfo.EvalGradHessian
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS
    # Used in compat.py
    num_accepted_steps: Int[Array, ""]


# TODO:
# - testing on a real benchmark problem
# - adding support for inequality constraints
# - trajectory optimisation example
# - iterative refinement
# - KKT error minimisation ahead of robust feasibility restoration
# - inertia correction
# - second-order correction
# - what happens to dual variables after feasibility restoration?
class AbstractIPOPTLike(
    AbstractMinimiser[Y, Aux, _IPOPTLikeState], Generic[Y, Aux], strict=True
):
    """Abstract IPOPT-like solver. Uses a filtered line search and an interior descent,
    and restores feasibility by solving a nonlinear subproblem if required. Approximates
    the Hessian using BFGS updates, as in [`optimistix.BFGS`][].

    This abstract version may be subclassed to choose alternative descent and searches.

    This solver will never evaluate the target or constraint functions outside the
    bounds placed on `y`, but it is somewhat geared toward equality constraints, and
    does not guarantee that these will not be violated. (Which is anyway not something
    that is really possible with nonlinear constraints.)

    If no bounds are provided, this solver defaults to infinite bounds on all elements
    of `y`. If the initial value `y0` is infeasible with respect to the bounds, it is
    moved to the strict interior of the bounded region.

    Note that this solver does not use its `rtol` attribute, only its `atol` attribute.
    # TODO: we follow IPOPT here, they also have only a single scalar tolerance.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    - `bound_push`: if the initial point `y0` is infeasible with respect to the bounds,
        it is moved to the interior of the feasible region. Default value is 0.01. This
        value should be positive, but we don't check whether it is here.
    - `bound_frac`: if the initial point `y0` is infeasible with respect to the bounds,
        it is moved to the interior of the feasible region. If upper and lower bounds
        are used, this value helps ensure that the point is not moved too close to the
        upper bound. Default value is 0.01. This value should not exceed 0.5, but this
        is not checked here.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, FunctionInfo.EvalGradHessian, Any]]
    search: AbstractVar[
        AbstractSearch[Y, FunctionInfo.EvalGradHessian, FunctionInfo.Eval, Any]
    ]
    verbose: AbstractVar[frozenset[str]]
    initial_barrier_parameter: AbstractVar[float]  # TODO: float or ScalarLike?

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _IPOPTLikeState:
        # TODO: Here we modify the initial value y0, to ensure that it is in the strict
        # interior with respect to the bound constraints. We then write this value into
        # the `iterate` attribute of the state. The way the solver is currently
        # constructed, this means that we will only evaluate the target function at
        # values that are in the strict interior of the bounded region. However, we
        # never actually overwrite `y`, which would require changing the return type of
        # solver.init to a tuple (y, state). So doing this here requires the assumption
        # that `y` is never used again.
        # Moving this functionality into `solver.step` is not preferable because the
        # introduction of clipping could mask bugs there, and should not be necessary in
        # this solver in particular.
        if bounds is None:
            bounds = (tree_full_like(y, -jnp.inf), tree_full_like(y, jnp.inf))
        lower, upper = bounds
        bound_push = options.get("bound_push", 0.01)
        bound_frac = options.get("bound_frac", 0.01)
        y = _interior_tree_clip(y, lower, upper, bound_push, bound_frac)

        if constraint is None:
            raise ValueError(
                "IPOPTLike requires constraints. For unconstrained problems, try "
                "an unconstrained minimiser, like `optx.BFGS`."
            )
        else:
            # TODO: this extra evaluation of the constraint can be expensive!
            # Not great for compilation time, we should avoid this here if we can.
            # (Not sure that we can, though.)
            evaluated = evaluate_constraint(constraint, y)
            constraint_residual, constraint_bound, constraint_jacobians = evaluated

        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        hessian = identity_pytree(y)
        f_info = FunctionInfo.EvalGradHessian(
            f,
            grad,
            hessian,
            y,
            bounds,
            constraint_residual,
            constraint_bound,
            constraint_jacobians,  # pyright: ignore - TODO: fix this
        )
        f_info_struct = eqx.filter_eval_shape(lambda: f_info)

        _, slack = constraint_residual
        slack_bounds = (tree_full_like(slack, 0.0), tree_full_like(slack, jnp.inf))
        slack = _interior_tree_clip(slack, *slack_bounds, bound_push, bound_frac)
        iterate = Iterate(
            y,
            slack,
            tree_full_like(constraint_residual, 0.0),
            tree_full_like(bounds, 1.0),
            jnp.asarray(self.initial_barrier_parameter),
        )

        return _IPOPTLikeState(
            first_step=jnp.array(True),
            iterate=iterate,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(iterate, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _IPOPTLikeState,
        tags: frozenset[object],
    ) -> tuple[Y, _IPOPTLikeState, Aux]:
        # TODO: support autodiff mode for the constraints too
        autodiff_mode = options.get("autodiff_mode", "bwd")
        if bounds is None:
            bounds = (tree_full_like(y, -jnp.inf), tree_full_like(y, jnp.inf))

        # TODO names! duals, boundary_multipliers? constraint_multipliers, bound_mult..?
        # y_eval, duals, bound_multipliers, barrier = state.iterate
        # Iterate with dummy values for the slack
        # dummy_slack = tree_full_like(duals[1], 0.0)
        # iterate = Iterate(y_eval, dummy_slack, duals, bound_multipliers, barrier)

        evaluated = evaluate_constraint(constraint, state.iterate.y_eval)
        constraint_residual, constraint_bound, constraint_jacobians = evaluated

        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.iterate.y_eval, has_aux=True
        )

        # TODO: with a second-order correction, all of these become proposed step sizes
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.iterate.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval, bounds, constraint_residual),
            state.search_state,
        )

        if not FILTERED_LINE_SEARCH:  # Mimic behavior of LearningRate(1.0)
            accept = jnp.array(True)
            step_size = jnp.array(1.0)
            search_result = RESULTS.successful
            search_state = state.search_state  # No update

        # TODO: this needs to accept an iterate, too - so we can return it, and also
        # modify the iterate in the rejected branch.
        # Right now working around this by having iterate, and iterate_
        def accepted(states):
            # jax.debug.print("accepted step")
            _, descent_state = states

            grad = lin_to_grad(
                lin_fn, state.iterate.y_eval, autodiff_mode=autodiff_mode
            )

            # TODO: WIP: Hessians of the Lagrangian
            hessian_, _ = jax.hessian(fn)(
                state.iterate.y_eval, args
            )  # no Hessian w.r.t. aux
            out_structure = jax.eval_shape(lambda: state.iterate.y_eval)
            hessian = lx.PyTreeLinearOperator(hessian_, out_structure)

            def constraint_hessian(y, jacobian_operator, multiplier):
                def apply_reduced_jacobian(_y):
                    mapped = jacobian_operator.T.mv(multiplier)
                    return jtu.tree_map(lambda a, b: a * b, mapped, _y)

                hessian = jax.jacfwd(apply_reduced_jacobian)(y)
                out_structure = jax.eval_shape(lambda: y)
                return lx.PyTreeLinearOperator(hessian, out_structure)

            equality_jacobian, inequality_jacobian = constraint_jacobians
            equality_multiplier, inequality_multiplier = state.iterate.multipliers
            equality_hessian = constraint_hessian(
                state.iterate.y_eval, equality_jacobian, equality_multiplier
            )
            inequality_hessian = constraint_hessian(
                state.iterate.y_eval, inequality_jacobian, inequality_multiplier
            )

            # TODO: work with adding the operators directly here? Does not work yet
            # because we initialise the Hessian in f_info with a specific structure
            # that composed linear operators do not have and with a positive
            # semidefinite tag that is also not appropriate here.

            lagrangian_hessian = jtu.tree_map(
                lambda a, b, c: a + b + c,
                hessian.pytree,
                equality_hessian.pytree,
                inequality_hessian.pytree,
            )
            lagrangian_hessian = lx.PyTreeLinearOperator(
                lagrangian_hessian,
                # (hessian_**ω).ω,
                jax.eval_shape(lambda: state.iterate.y_eval),
                lx.positive_semidefinite_tag,  # TODO Not technically correct!
            )
            f_eval_info_ = FunctionInfo.EvalGradHessian(
                f_eval,
                grad,
                lagrangian_hessian,
                state.iterate.y_eval,
                bounds,
                constraint_residual,
                constraint_bound,
                constraint_jacobians,  # pyright: ignore
            )

            # TODO: something going on here with the permitted types, fixing this is
            # punted until we make a decision on whether to unify termination criteria
            # with a common interface
            # TODO: update to structured iterate
            # TODO: include slack in termination condition
            iterate_ = (
                state.iterate.y_eval,
                state.iterate.multipliers,
                state.iterate.bound_multipliers,
                state.iterate.barrier,
            )
            error = _error(
                iterate_,
                f_eval_info_,  # pyright: ignore
                0.01,  # This is supposed to be the barrier parameter
                self.atol,
                self.norm,  # pyright: ignore
            )
            # TODO: we're introducing new hyperparameters here!
            converged_at_barrier = error <= 10 * state.iterate.barrier
            new_barrier = jnp.min(
                jnp.array([0.2 * state.iterate.barrier, state.iterate.barrier**1.5])
            )
            new_barrier = jnp.max(jnp.array([new_barrier, self.atol / 10]))
            new_barrier = jnp.where(
                converged_at_barrier, new_barrier, state.iterate.barrier
            )
            # TODO: I cannot return this barrier update unless I return it through an
            # iterate.

            terminate = error <= self.atol
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)

            descent_state = self.descent.query(
                state.iterate,
                f_eval_info_,  # pyright: ignore
                descent_state,
            )

            return (
                state.iterate.y_eval,  # TODO: return iterate?
                f_eval_info_,
                aux_eval,
                search_state,
                descent_state,
                terminate,
                new_barrier,
            )

        def rejected(states):
            search_state, descent_state = states

            # if SECOND_ORDER_CORRECTION:
            #     updated_f_info = eqx.tree_at(
            #         lambda f: f.constraint_residual,
            #         state.f_info,
            #         (state.f_info.constraint_residual**ω + constraint_residual**ω).ω,
            #     )
            #     descent_state = self.descent.query(
            #         state.iterate,
            #         updated_f_info,
            #         descent_state,
            #     )

            # TODO: SOC with twice the step size as is returned by the first search...
            # Alternatively calling again with state.search_state. This means that we
            # lose the filter augmentation from the first call to the search. We'd like
            # to keep the filter, I think - but this comes at the cost of making
            # assumptions about how much the search is backtracking.

            return (
                y,
                state.f_info,
                state.aux,
                search_state,
                descent_state,
                jnp.array(False),
                state.iterate.barrier,  # No update to barrier parameter
            )

        # Branch for normal acceptance / rejection: but rejection also handles SOC
        # TODO: barrier currently unused?
        y, f_info, aux, search_state, descent_state, terminate, barrier = filter_cond(
            accept, accepted, rejected, (search_state, state.descent_state)
        )

        if len(self.verbose) > 0:
            verbose_loss = "loss" in self.verbose
            verbose_step_size = "step_size" in self.verbose
            verbose_y = "y" in self.verbose
            loss_eval = f_eval
            loss = state.f_info.f
            verbose_print(
                (verbose_loss, "Loss on this step", loss_eval),
                (verbose_loss, "Loss on the last accepted step", loss),
                (verbose_step_size, "Step size", step_size),
                (verbose_y, "y", state.iterate.y_eval),
                (verbose_y, "y on the last accepted step", y),
            )

        descent_steps, descent_result = self.descent.step(step_size, descent_state)
        requires_restoration = (
            search_result == RESULTS.feasibility_restoration_required
        ) | (descent_result == RESULTS.feasibility_restoration_required)

        def restore(args):
            jax.debug.print("restoring feasibility")
            del args

            # TODO: make attribute and update the penalty parameter for the feasibility
            # restoration problem based on the barrier parameter.
            boundary_map = ClosestFeasiblePoint(1e-6, BFGS_B(rtol=1e-3, atol=1e-6))
            recovered_y, restoration_result = boundary_map(
                state.iterate.y_eval, constraint, bounds
            )
            # TODO: Allow feasibility restoration to raise a certificate of
            # infeasibility and error out.

            # TODO: perhaps re-set the dual variables here? So that we could combine
            # this solver with a descent that does not implement a least-squares
            # initialisation of the dual variables.
            iterate_eval = (state.iterate**ω + descent_steps**ω).ω
            # TODO: barrier update? This takes barrier from outer scope
            # What happens to the barrier parameter when we restore feasibility?
            # TODO: re-evaluate the constraint function to update the slack?
            # We're currently losing information by re-initialising the slack value
            # in the descent, which then does not get taken into account when updating
            # the slack variable! (I think? Does the slack get returned by the
            # initialisation method for the multipliers?)

            # TODO: now we're calling expensive functions again! not great
            _, slack = constraint(recovered_y)  # pyright: ignore (constraint not None)
            slack_bounds = (tree_full_like(slack, 0.0), tree_full_like(slack, jnp.inf))
            slack = _interior_tree_clip(slack, *slack_bounds, 0.01, 0.01)

            new_iterate = Iterate(
                recovered_y,
                slack,
                iterate_eval.multipliers,
                iterate_eval.bound_multipliers,
                barrier,
            )

            # Re-initialise the search
            f_info_struct = eqx.filter_eval_shape(lambda: state.f_info)
            new_search_state = self.search.init(recovered_y, f_info_struct)

            # Descent can special-case the first step it takes, e.g. to initialise dual
            # variables with a least-squares estimate. To enable this, we re-initialise
            # the descent state here. (Alternative: make descents take a first step
            # argument, like the searches do.)
            new_descent_state = self.descent.init(new_iterate, f_info_struct)

            new_solver_state = _IPOPTLikeState(
                first_step=jnp.array(True),
                iterate=new_iterate,
                search_state=new_search_state,
                f_info=state.f_info,
                aux=state.aux,
                descent_state=new_descent_state,
                terminate=jnp.array(False),
                result=restoration_result,
                num_accepted_steps=state.num_accepted_steps,
            )

            return new_solver_state

        # TODO: do we still do the right thing? Now its not too clear to me, given that
        # we do accept a y that is not part of the iterate structure.
        def regular_update(args):
            search_result, descent_result = args
            result = RESULTS.where(
                search_result == RESULTS.successful, descent_result, search_result
            )
            iterate_eval = (state.iterate**ω + descent_steps**ω).ω

            # TODO: remove this hack once the accepted branch returns an iterate
            iterate_eval = eqx.tree_at(lambda i: i.barrier, iterate_eval, barrier)

            return _IPOPTLikeState(
                first_step=jnp.array(False),
                iterate=iterate_eval,
                search_state=search_state,
                f_info=f_info,
                aux=aux_eval,
                descent_state=descent_state,
                terminate=terminate,
                result=result,
                num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
            )

        args = (search_result, descent_result)
        if not FEASIBILITY_RESTORATION:  # Disable during debugging of other features
            requires_restoration = jnp.array(False)
        state = filter_cond(requires_restoration, restore, regular_update, args)

        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _IPOPTLikeState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _IPOPTLikeState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        jax.debug.print("final barrier: {}", state.iterate.barrier)
        return y, aux, {}


# TODO: name of descent
# TODO: Edit docstring - this needs to be expanded quite a bit
class IPOPTLike(AbstractIPOPTLike[Y, Aux], strict=True):
    """An IPOPT-like solver. Uses a filtered line search and an interior descent, and
    restores infeasible steps by solving a nonlinear subproblem if required.

    Approximates the Hessian using BFGS updates, as in [`optimistix.BFGS`][].

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: IPOPTLikeDescent
    search: IPOPTLikeFilteredLineSearch
    verbose: frozenset[str]
    initial_barrier_parameter: ScalarLike

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
        initial_barrier_parameter: ScalarLike = 0.1,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = IPOPTLikeDescent()
        self.search = IPOPTLikeFilteredLineSearch()
        self.verbose = verbose
        self.initial_barrier_parameter = initial_barrier_parameter


IPOPTLike.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example 
    `verbose=frozenset({"step_size", "loss"})`.
- `initial_barrier_parameter`: The initial value of the barrier parameter. 
"""
