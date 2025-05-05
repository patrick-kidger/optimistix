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
from .bfgs import identity_pytree
from .filtered import IPOPTLikeFilteredLineSearch
from .interior import NewInteriorDescent


# TODO: conceptually the tolerance here is closer to an rtol, so we should probably
# change it to that (currently we only use atol)
# TODO: test what happens if inequality constraints have infinite bounds
# TODO: I observe that during the solve for some of the benchmark problems, a lot of
# tiny steps are taken (even if all steps are accepted), until suddently a large step is
# taken. This could bode well for second-order corrections! Maybe they would help on
# these problems. (HATFLDH comes to mind.)


# Some global flags strictly for use during development, these will be removed later.
# I'm introducing them here to be able to selectively enable certain special features,
# for use in testing and debugging.
# Note that it generally does not make sense to enable the filtered line search without
# enabling the feasibility restoration too, since this is currently the only place that
# the filter gets re-set. (We don't support the heuristic filter reset in the search
# yet.)
SECOND_ORDER_CORRECTION = False
FEASIBILITY_RESTORATION = False
FILTERED_LINE_SEARCH = False


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
    msg = (
        "The bounds for `y` are misspecified: `lower` must be strictly less than "
        "`upper` for all elements of the PyTree, since the feasible region otherwise "
        "does not have a strict interior."
    )
    proper_interval = jtu.tree_map(lambda l, u: jnp.all(l < u), lower, upper)
    interval_checks, _ = jfu.ravel_pytree(proper_interval)
    pred = jnp.all(interval_checks)
    lower, upper = eqx.error_if((lower, upper), jnp.invert(pred), msg)

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

    moved_lower = (lower**ω + lower_offset**ω).ω
    moved_upper = (upper**ω - upper_offset**ω).ω

    return tree_clip(tree, moved_lower, moved_upper)


# TODO: After feasibility restoration (?) IPOPT apparently sets all multipliers to zero
# IIUC: https://coin-or.github.io/Ipopt/OPTIONS.html
# Should we do this here? Then we would need to set separate cutoff values for the
# initial and subsequent initialisations of the multipliers.
# They now also support an option to set the bound multipliers to mu/value, as we do
# below for the slack variables.
# ...and it turns out that they do their linear least squares solve for all multipliers,
# not just the equality multipliers. So this would require a change here. I'm tabling
# this for now though, since its not clear that it makes this much of a difference and
# other pots are burning, to use a german phrase. The inequality multipliers are also
# restricted with respect to the values they may take (currently strictly negative), so
# this would require a non-negative least squares solve for them, or some other kind of
# correction.
def _initialise_multipliers(iterate__f_info):
    """Initialise the multipliers for the equality and inequality constraints. Both are
    initialised to the value they are expected to take at the optimum, with a safety
    factor for truncation if the computed values are unexpectedly large.

    The multipliers for the inequality constraints are initialised to the value they are
    expected to take at the optimum, which is

        inequality_multiplier = barrier_parameter / slack

    where the slack variables convert the inequality constraints to equality constraints
    such that we have

        g(y) - slack = 0.

    For an inequality constraint function `g`. The slack variables must always be
    strictly positive.

    We can then solve for the multipliers of the equality constraints with a linear
    least squares solve, again assuming optimality of the Lagrangian, which means that

        grad Lagrangian = grad f + Jac h^T * l + Jac g^T * m = 0

    for an objective function `f`, equality constraints `h`, and inequality constraints
    `g`, with inequality constraint multipliers `m` computed as above. We then get

        Jac h^T * l = -grad f - Jac g^T * m

    which we can solve for the equality constraint multipliers `l`. We can do better
    than that though, since we know that the role of the Lagrangian multipliers is to
    counterbalance the gradient of the objective function at the optimum. This means
    that directions of the "constraint gradient" Jac h^T * l that are orthogonal to the
    gradient of the objective function can be discarded.
    To accomplish this, we introduce an auxiliary variable `w` that we require to be in
    the null space of the Jacobian of the equality constraints. We obtain the linear
    system

        [     I    Jac h^T ] [ w ] = [ -grad f - Jac g^T * m]
        [ Jac h^T    0     ] [ l ] = [          0           ]

    which we solve by least squares.

    Finally, strong linear dependence of the equality constraints can result in very
    large values for the multipliers computed in this way. To guard against this, we
    truncate all multipliers to a cutoff value, including the multipliers for the
    inequality constraints.

    Note: (TODO): IPOPT supports separate options for the `bound_frac` and `bound_push`
    parameters of the interior clipping function. Right now we do not support this and
    instead use a (hard-coded) value of 0.01 for both. The respective values are set as
    solver options, so patching them through to the descent would need to be figured
    out. Alternatively the value of the barrier parameter could be used, to make sure
    that we move the slack variables less as the overall solve progresses. However, this
    somewhat contradicts the purpose of the initialisation here - which should be
    robust, since it is done at the start and after every feasibility restoration step.
    """
    iterate, f_info = iterate__f_info
    (equality_multipliers, _) = iterate.multipliers
    _, inequality_residual = f_info.constraint_residual

    # TODO: special case for when we don't have any inequality constraints
    # TODO: this is currently a bit duplicate with code elsewhere! We initialise a slack
    # variable here, but this might not be the best idea - this should be done in one
    # place. And that place is probably not here.
    # If I'm not initialising the slack variables here, then I also do not need to have
    # this function be in the know about the truncation to the strict interior.
    # (So we can remove the hard-coded parameters below.)
    slack = inequality_residual
    lower = tree_full_like(slack, 0.0)
    upper = tree_full_like(slack, jnp.inf)
    slack = _interior_tree_clip(slack, lower, upper, 0.01, 0.01)
    inequality_multipliers = (-iterate.barrier / slack**ω).ω
    # jax.debug.print("slack: {}", slack)
    # jax.debug.print("inequality multipliers: {}", inequality_multipliers)

    equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
    inequality_gradient = inequality_jacobian.T.mv(inequality_multipliers)
    gradient = (-(f_info.grad**ω) - inequality_gradient**ω).ω
    null = tree_full_like(equality_multipliers, 0.0)
    vector = (gradient, null)
    del equality_multipliers

    def make_operator(equality_jacobian, input_structure):
        jac = equality_jacobian  # for brevity

        def operator(inputs):
            orthogonal_component, parallel_component = inputs
            r1 = (orthogonal_component**ω + jac.T.mv(parallel_component) ** ω).ω
            r2 = jac.mv(orthogonal_component)
            return r1, r2

        return lx.FunctionLinearOperator(operator, input_structure)

    operator = make_operator(equality_jacobian, jax.eval_shape(lambda: vector))
    # TODO: we could allow a different choice of linear solver here
    out = lx.linear_solve(operator, vector, lx.SVD())
    _, equality_multipliers = out.value

    # TODO: hard-coded cutoff value for now! In IPOPT this is an option to the solver
    def truncate(x):
        return jnp.where(jnp.abs(x) > 1e3, x, 0.0)

    safe_equality_multipliers = jtu.tree_map(truncate, equality_multipliers)
    safe_inequality_multipliers = jtu.tree_map(truncate, inequality_multipliers)
    safe_duals = (safe_equality_multipliers, safe_inequality_multipliers)

    return Iterate(
        iterate.y_eval,
        iterate.slack,  # TODO: we're not currently updating the slack variables here!
        safe_duals,  # Only multipliers were updated
        iterate.bound_multipliers,
        iterate.barrier,
    )


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
    y, multipliers, bound_multipliers, barrier = iterate  # TODO structured iterate

    # Construction site: lagrangian gradients ------------------------------------------
    # We need to have a specific module for Lagrangian utilities, I think!
    lagrangian_gradient = f_info.grad  # TODO reuse this! We have a function for this
    if f_info.constraint_jacobians is not None:
        equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
        equality_multiplier, inequality_multiplier = multipliers
        if equality_jacobian is not None:
            equality_gradient = equality_jacobian.T.mv(equality_multiplier)
            lagrangian_gradient = (lagrangian_gradient**ω + equality_gradient**ω).ω
        if inequality_jacobian is not None:
            inequality_gradient = inequality_jacobian.T.mv(inequality_multiplier)
            lagrangian_gradient = (lagrangian_gradient**ω + inequality_gradient**ω).ω
    if f_info.bounds is not None:
        lb_dual, ub_dual = bound_multipliers  # TODO name
        lower, upper = f_info.bounds
        finite_lower = jtu.tree_map(jnp.isfinite, lower)
        finite_upper = jtu.tree_map(jnp.isfinite, upper)
        lb_dual = tree_where(finite_lower, lb_dual, 0.0)  # Safety: should be redundant
        ub_dual = tree_where(finite_upper, ub_dual, 0.0)
        lagrangian_gradient = (lagrangian_gradient**ω - lb_dual**ω + ub_dual**ω).ω
    optimality_error = norm(lagrangian_gradient)
    # CONSTRUCTION SITE ends: Lagrangian gradients
    # ----------------------------------------------------------------------------------

    # ANOTHER CONSTRUCTION SITE BEGINS: ------------------------------------------------
    # Constraint norms
    constraint_norm = jnp.array(0.0)  # Should not affect max if no constraints
    if f_info.constraint_residual is not None:
        equality_residual, inequality_residual = f_info.constraint_residual
        if equality_residual is not None:
            # TODO: implement support for inequality residuals
            equality_residual, inequality_residual = f_info.constraint_residual
            constraint_norm += norm(equality_residual)
        if inequality_residual is not None:
            inequality_violation = tree_where(
                # Only count violations if the residual is less than zero
                # Alternatively transform with slack! # TODO
                jtu.tree_map(  # TODO use slacks here?
                    lambda x: jnp.where(x < 0, x, 0.0), inequality_residual
                ),
                inequality_residual,
                0.0,
            )
            constraint_norm += norm(inequality_violation)
    # CONSTRUCTION SITE ENDS: ----------------------------------------------------------

    ### TO BE REFACTORED (made a separate optx.one_norm) -------------------------------
    def _one_norm(x):
        absolute_values = jnp.where(jnp.isfinite(x), jnp.abs(x), 0.0)
        return jnp.sum(absolute_values)

    if multipliers is not None:
        multiplier_norm = jtu.tree_map(_one_norm, multipliers)
        multiplier_norm, _ = jfu.ravel_pytree(multiplier_norm)
        multiplier_norm = jnp.sum(multiplier_norm)

        num_multipliers = jtu.tree_map(jnp.isfinite, multipliers)
        num_multipliers, _ = jfu.ravel_pytree(num_multipliers)
        num_multipliers = jnp.sum(num_multipliers)
    else:
        multiplier_norm = jnp.array(0.0)
        num_multipliers = jnp.array(0.0)  # Needs safe division down below

    if bound_multipliers is not None:
        bound_multiplier_norm = jtu.tree_map(_one_norm, bound_multipliers)
        bound_multiplier_norm, _ = jfu.ravel_pytree(bound_multiplier_norm)
        bound_multiplier_norm = jnp.sum(bound_multiplier_norm)

        num_bounds = jtu.tree_map(jnp.isfinite, bound_multipliers)  # Or check bounds
        num_bounds, _ = jfu.ravel_pytree(num_bounds)
        num_bounds = jnp.sum(num_bounds)
    else:
        bound_multiplier_norm = jnp.array(0.0)
        num_bounds = jnp.array(0.0)
    # ----------------------------------------------------------------------------------

    # TODO: This scaling thing should also be moved I think -- and it smax now hard-code
    summed_norms = multiplier_norm + bound_multiplier_norm
    denominator = num_multipliers + num_bounds
    safe_denominator = jnp.where(denominator > 1.0, denominator, 1.0)
    summed_scaled_norm = summed_norms / safe_denominator
    scaling = jnp.where(summed_scaled_norm > 100.0, summed_scaled_norm, 100.0) / 100.0

    optimality_error = optimality_error / scaling

    # Aggregate errata - probably move this for better flow
    errata = (optimality_error, constraint_norm)

    if f_info.bounds is not None:
        lower, upper = f_info.bounds  # pyright: ignore (bounds not None)  # TODO
        lower_diff = (y**ω - lower**ω).ω
        upper_diff = (upper**ω - y**ω).ω
        lb_dual, ub_dual = bound_multipliers
        lower_error = (lower_diff**ω * lb_dual**ω - barrier).ω
        upper_error = (upper_diff**ω * ub_dual**ω - barrier).ω
        finite_lower = jtu.tree_map(jnp.isfinite, lower)
        finite_upper = jtu.tree_map(jnp.isfinite, upper)
        lower_error = tree_where(finite_lower, lower_error, 0.0)
        upper_error = tree_where(finite_upper, upper_error, 0.0)

        # TODO refactor: scaling the bound errata --------------------------------------
        finite_lower, _ = jfu.ravel_pytree(finite_lower)
        finite_upper, _ = jfu.ravel_pytree(finite_upper)
        num_lower = jnp.sum(finite_lower)
        num_upper = jnp.sum(finite_upper)

        lower_bound_scale = jnp.where(
            num_lower > 0, _one_norm(lb_dual) / num_lower, 0.0
        )
        upper_bound_scale = jnp.where(
            num_upper > 0, _one_norm(ub_dual) / num_upper, 0.0
        )

        lower_bound_scale = (
            jnp.where(lower_bound_scale > 100.0, lower_bound_scale, 100.0) / 100.0
        )
        upper_bound_scale = (
            jnp.where(upper_bound_scale > 100.0, upper_bound_scale, 100.0) / 100.0
        )

        lower_error_ = norm(lower_error) / lower_bound_scale
        upper_error_ = norm(upper_error) / upper_bound_scale

        errata += (lower_error_, upper_error_)

    return jnp.max(jnp.asarray(errata))


class Iterate(eqx.Module, Generic[Y, EqualityOut, InequalityOut], strict=True):
    y_eval: Y  # TODO: or more concisely rename y?
    slack: InequalityOut | None
    multipliers: tuple[EqualityOut, InequalityOut]
    bound_multipliers: tuple[Y, Y] | None
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
        bound_push = options.get("bound_push", 0.01)
        bound_frac = options.get("bound_frac", 0.01)

        if bounds is not None:
            lower, upper = bounds
            y = _interior_tree_clip(y, lower, upper, bound_push, bound_frac)

        if constraint is not None:
            # TODO: this extra evaluation of the constraint can be expensive!
            # Not great for compilation time, we should avoid this here if we can.
            # (Not sure that we can, though.)
            evaluated = evaluate_constraint(constraint, y)
            constraint_residual, constraint_bound, constraint_jacobians = evaluated

            _, inequality_residual = constraint_residual
            if inequality_residual is not None:
                slack = inequality_residual
                slack_bounds = (
                    tree_full_like(slack, 0.0),
                    tree_full_like(slack, jnp.inf),
                )
                slack = _interior_tree_clip(
                    slack, *slack_bounds, bound_push, bound_frac
                )
            else:
                slack = None
        else:
            constraint_residual = constraint_bound = constraint_jacobians = None
            slack = None

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

        # CONSTRUCTION SITE: Initialisation of bound multipliers -----------------------
        if bounds is not None:
            lower, upper = bounds
            lower_multipliers = tree_full_like(lower, 1.0)
            upper_multipliers = tree_full_like(upper, 1.0)
            finite_lower = jtu.tree_map(jnp.isfinite, lower)
            finite_upper = jtu.tree_map(jnp.isfinite, upper)
            lower_multipliers = tree_where(finite_lower, lower_multipliers, 0.0)
            upper_multipliers = tree_where(finite_upper, upper_multipliers, 0.0)
            bound_multipliers = (lower_multipliers, upper_multipliers)
        else:
            bound_multipliers = None

        iterate = Iterate(
            y,
            slack,
            tree_full_like(constraint_residual, 0.0),
            bound_multipliers,  # pyright: ignore
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

        # TODO names! duals, boundary_multipliers? constraint_multipliers, bound_mult..?
        # y_eval, duals, bound_multipliers, barrier = state.iterate
        # Iterate with dummy values for the slack
        # dummy_slack = tree_full_like(duals[1], 0.0)
        # iterate = Iterate(y_eval, dummy_slack, duals, bound_multipliers, barrier)

        if constraint is not None:
            evaluated = evaluate_constraint(constraint, state.iterate.y_eval)
            constraint_residual, constraint_bound, constraint_jacobians = evaluated
        else:
            constraint_residual = constraint_bound = constraint_jacobians = None

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
            _, descent_state = states

            grad = lin_to_grad(
                lin_fn, state.iterate.y_eval, autodiff_mode=autodiff_mode
            )

            # TODO: WIP: Hessians of the Lagrangian
            lagrangian_hessian, _ = jax.hessian(fn)(
                state.iterate.y_eval, args
            )  # no Hessian w.r.t. aux

            def constraint_hessian(y, jacobian_operator, multiplier):
                def apply_reduced_jacobian(_y):
                    mapped = jacobian_operator.T.mv(multiplier)
                    return jtu.tree_map(lambda a, b: a * b, mapped, _y)

                return jax.jacfwd(apply_reduced_jacobian)(y)

            if constraint_jacobians is not None:
                equality_jacobian, inequality_jacobian = constraint_jacobians
                equality_multiplier, inequality_multiplier = state.iterate.multipliers
                if equality_jacobian is not None:
                    equality_hessian = constraint_hessian(
                        state.iterate.y_eval, equality_jacobian, equality_multiplier
                    )
                    lagrangian_hessian = (lagrangian_hessian**ω + equality_hessian**ω).ω
                if inequality_jacobian is not None:
                    inequality_hessian = constraint_hessian(
                        state.iterate.y_eval, inequality_jacobian, inequality_multiplier
                    )
                    lagrangian_hessian = (
                        lagrangian_hessian**ω + inequality_hessian**ω
                    ).ω

            # TODO: work with adding the operators directly here? Does not work yet
            # because we initialise the Hessian in f_info with a specific structure
            # that composed linear operators do not have and with a positive
            # semidefinite tag that is also not appropriate here.

            lagrangian_hessian = lx.PyTreeLinearOperator(
                lagrangian_hessian,
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

            if SECOND_ORDER_CORRECTION:
                updated_f_info = eqx.tree_at(
                    lambda f: f.constraint_residual,
                    state.f_info,
                    (state.f_info.constraint_residual**ω + constraint_residual**ω).ω,
                )
                _previous_step = descent_state.step
                descent_state_ = self.descent.query(
                    state.iterate,
                    updated_f_info,
                    descent_state,
                )
                _corrected_step = descent_state_.step
                _step_diff = (_corrected_step**ω - _previous_step**ω).ω
                del _step_diff

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
            search_result, descent_result = args
            # del args

            # # TODO: make attribute and update the penalty parameter for the feasibilit
            # # restoration problem based on the barrier parameter.
            # boundary_map = ClosestFeasiblePoint(1e-6, BFGS_B(rtol=1e-3, atol=1e-6))
            # recovered_y, restoration_result = boundary_map(
            #     state.iterate.y_eval, constraint, bounds
            # )
            # # TODO: Allow feasibility restoration to raise a certificate of
            # # infeasibility and error out.

            # # TODO: perhaps re-set the dual variables here? So that we could combine
            # # this solver with a descent that does not implement a least-squares
            # # initialisation of the dual variables.
            iterate_eval = (state.iterate**ω + descent_steps**ω).ω
            # # TODO: barrier update? This takes barrier from outer scope
            # # What happens to the barrier parameter when we restore feasibility?
            # # TODO: re-evaluate the constraint function to update the slack?
            # # We're currently losing information by re-initialising the slack value
            # # in the descent, which then does not get taken into account when updating
            # # the slack variable! (I think? Does the slack get returned by the
            # # initialisation method for the multipliers?)

            # # TODO: now we're calling expensive functions again! not great
            # _, slack = constraint(recovered_y)  # pyright: ignore (constraint not None
            # slack_bounds = (tree_full_like(slack, 0.0), tree_full_like(slack, jnp.inf)
            # slack = _interior_tree_clip(slack, *slack_bounds, 0.01, 0.01)

            # new_iterate = Iterate(
            #     recovered_y,
            #     slack,
            #     iterate_eval.multipliers,
            #     iterate_eval.bound_multipliers,
            #     barrier,
            # )

            # # Re-initialise the search
            # f_info_struct = eqx.filter_eval_shape(lambda: state.f_info)
            # new_search_state = self.search.init(recovered_y, f_info_struct)

            # # Descent can special-case the first step it takes, e.g. to initialise dua
            # # variables with a least-squares estimate. To enable this, we re-initialis
            # # the descent state here. (Alternative: make descents take a first step
            # # argument, like the searches do.)
            # new_descent_state = self.descent.init(new_iterate, f_info_struct)

            result = RESULTS.where(
                search_result == RESULTS.successful, descent_result, search_result
            )

            # TODO: completely turned off feasibility restoration for now
            new_solver_state = _IPOPTLikeState(
                first_step=jnp.array(True),
                iterate=iterate_eval,
                search_state=search_state,
                f_info=state.f_info,
                aux=state.aux,
                descent_state=descent_state,
                terminate=jnp.array(False),
                result=result,
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
    descent: NewInteriorDescent
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
        self.descent = NewInteriorDescent()
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
