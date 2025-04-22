from collections.abc import Callable
from typing import Any, Generic, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

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
from .bfgs import BFGS, identity_pytree
from .boundary_maps import ClosestFeasiblePoint
from .filtered import FilteredLineSearch


# Some global flags strictly for use during development, these will be removed later.
# I'm introducing them here to be able to selectively enable certain special features,
# for use in testing and debugging.
SECOND_ORDER_CORRECTION = False


def _make_kkt_operator(hessian, jacobian, input_structure):
    def kkt(inputs):
        y, duals = inputs
        y_step = (hessian.mv(y) ** ω + jacobian.T.mv(duals) ** ω).ω
        dual_step = jacobian.mv(y)
        return y_step, dual_step

    return lx.FunctionLinearOperator(kkt, input_structure)


def _make_barrier_hessians(
    y: Y,
    bounds: tuple[Y, Y],
    bound_multipliers: tuple[Y, Y],
) -> tuple[lx.DiagonalLinearOperator, lx.DiagonalLinearOperator]:
    """Construct the "barrier Hessians" for the logarithmic barrier terms in the merit
    function. They are constructed as a diagonal matrix to be added to the Hessian of
    the target function, composed of the reciprocals of the distances to finite bounds,
    multiplied by the Lagrange multipliers for the bound constraints.
    """
    assert jtu.tree_structure(bound_multipliers) == jtu.tree_structure(bounds)

    lower, upper = bounds
    finite_lower = jtu.tree_map(jnp.isfinite, lower)
    finite_upper = jtu.tree_map(jnp.isfinite, upper)

    mul_lower, mul_upper = bound_multipliers
    # We're adding a safeguard here to avoid special-casing the first step.
    mul_lower = tree_where(finite_lower, mul_lower, 0.0)
    mul_upper = tree_where(finite_upper, mul_upper, 0.0)

    _ones = tree_full_like(y, 1.0)
    lower_term = tree_where(finite_lower, (1 / (y**ω - lower**ω)).ω, _ones)
    upper_term = tree_where(finite_upper, (1 / (upper**ω - y**ω)).ω, _ones)

    lower_hessian = lx.DiagonalLinearOperator((lower_term**ω * mul_lower**ω).ω)
    upper_hessian = lx.DiagonalLinearOperator((upper_term**ω * mul_upper**ω).ω)

    return lower_hessian, upper_hessian


def _make_barrier_gradients(
    y: Y, bounds: tuple[Y, Y], barrier_parameter: float
) -> tuple[Y, Y]:
    """Define the gradient of the merit function. This is the log-barrier gradient,
    defined as the sum of the gradients of the logarithmic barrier terms for the bound
    constraints.
    """
    lower, upper = bounds
    finite_lower = jtu.tree_map(jnp.isfinite, lower)
    finite_upper = jtu.tree_map(jnp.isfinite, upper)
    dummy_barrier_parameter = 0.01
    lower_barrier_grad = (dummy_barrier_parameter / (y**ω - lower**ω)).ω
    upper_barrier_grad = (dummy_barrier_parameter / (upper**ω - y**ω)).ω
    lower_barrier_grad = tree_where(finite_lower, lower_barrier_grad, 0.0)
    upper_barrier_grad = tree_where(finite_upper, upper_barrier_grad, 0.0)
    return lower_barrier_grad, upper_barrier_grad


class _IPOPTLikeDescentState(
    eqx.Module, Generic[Y, EqualityOut, InequalityOut], strict=True
):
    step: PyTree
    result: RESULTS


# TODO: perhaps streamline this - if I have an iterate that carries a formalism on how
# its constraints and bounds should be handled.
class IPOPTLikeDescent(
    AbstractDescent[Y, FunctionInfo.EvalGradHessian, _IPOPTLikeDescentState],
    strict=True,
):
    """A descent method, closely inspired by IPOPT. Regularises the KKT system before
    taking a Newton step.

    Requires bounds on all elements of `y`, which may be infinite.
    [`optimistix.IPOPTLike`][] defaults to infinite bounds if none are provided. To use
    this descent with another solver, we recommend the same approach.
    """

    linear_solver: lx.AbstractLinearSolver = lx.SVD()

    def init(  # pyright: ignore (figuring out what types a descent should take)
        self, iterate, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _IPOPTLikeDescentState:
        if f_info_struct.bounds is None:
            raise ValueError(
                "IPOPTLikeDescent requires bounds on the optimisation variable `y`. "
                "To use this descent without bound constraints, pass infinite bounds "
                "on all elements of `y` instead of specifying `bounds=None`."
            )

        if f_info_struct.constraint_residual is None:
            raise ValueError  # TODO better errors
        else:
            return _IPOPTLikeDescentState(iterate, RESULTS.successful)

    def query(  # pyright: ignore  (figuring out what types a descent should take)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _IPOPTLikeDescentState,
    ) -> _IPOPTLikeDescentState:
        y, (equality_dual, inequality_dual), boundary_multipliers = iterate

        assert f_info.constraint_residual is not None
        assert f_info.constraint_jacobians is not None

        barrier_hessians = _make_barrier_hessians(
            y,
            f_info.bounds,  # pyright: ignore
            boundary_multipliers,  # pyright: ignore
        )
        lower_barrier_hessian, upper_barrier_hessian = barrier_hessians
        hessian = f_info.hessian + lower_barrier_hessian + upper_barrier_hessian
        barrier_grads = _make_barrier_gradients(y, f_info.bounds, 0.01)  # pyright: ignore
        lower_barrier_grad, upper_barrier_grad = barrier_grads
        grad = (f_info.grad**ω - lower_barrier_grad**ω - upper_barrier_grad**ω).ω

        equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
        equality_residual, inequality_residual = f_info.constraint_residual
        input_structure = jax.eval_shape(lambda: (y, equality_residual))
        kkt_operator = _make_kkt_operator(hessian, equality_jacobian, input_structure)

        dual_term = equality_jacobian.T.mv(equality_dual)  # pyright: ignore  # TODO
        vector = (
            (-(grad**ω) - dual_term**ω).ω,
            (-(equality_residual**ω)).ω,
        )
        out = lx.linear_solve(kkt_operator, vector, self.linear_solver)

        y_step, eqdual_step = out.value
        result = RESULTS.promote(out.result)

        def get_boundary_steps(
            multiplier, barrier_gradient, barrier_hessian, y_step, bound
        ):
            step = (
                barrier_gradient**ω - multiplier**ω - barrier_hessian.mv(y_step) ** ω
            ).ω
            finite_bound = jtu.tree_map(jnp.isfinite, bound)
            return tree_where(finite_bound, step, 0.0)

        lower_bound_dual, upper_bound_dual = boundary_multipliers  # pyright: ignore
        lower, upper = f_info.bounds  # pyright: ignore
        lower_mstep = get_boundary_steps(
            lower_bound_dual,
            lower_barrier_grad,
            lower_barrier_hessian,
            y_step,
            lower,
        )
        upper_mstep = get_boundary_steps(
            upper_bound_dual,
            upper_barrier_grad,
            upper_barrier_hessian,
            y_step,
            upper,
        )
        b_steps = (lower_mstep, upper_mstep)

        offset = 0.01  # Standard in IPOPT (tau_min)
        feasible_lower_m_step_size = feasible_step_length(
            lower_bound_dual, tree_full_like(y, 0.0), lower_mstep, offset=offset
        )
        feasible_upper_m_step_size = feasible_step_length(
            upper_bound_dual, tree_full_like(y, 0.0), upper_mstep, offset=offset
        )
        lower_m_step = (feasible_lower_m_step_size * lower_mstep**ω).ω
        upper_m_step = (feasible_upper_m_step_size * upper_mstep**ω).ω
        b_steps = (lower_m_step, upper_m_step)

        # I now need to compute a "slack step", i.e. define the distance to the
        # bounds. Both are positive numbers
        lower, upper = f_info.bounds  # pyright: ignore
        lower_max_step_size = feasible_step_length(y, lower, y_step, offset=offset)
        upper_max_step_size = feasible_step_length(y, upper, y_step, offset=offset)

        max_step_size = jnp.min(jnp.array([lower_max_step_size, upper_max_step_size]))

        y_step = (max_step_size * y_step**ω).ω
        eqdual_step = (max_step_size * eqdual_step**ω).ω

        dummy = inequality_residual  # Not yet used or updated
        iterate_step = (y_step, (eqdual_step, dummy), b_steps)

        return _IPOPTLikeDescentState(
            iterate_step,
            result,
        )

    def step(
        self, step_size: Scalar, state: _IPOPTLikeDescentState
    ) -> tuple[Y, RESULTS]:
        # TODO Note that I *am* currently scaling the dual variables for the bounds too
        # Fixing this would require unpacking the tuple and packing it up again
        # When very large values for the bounds are used, it does seem restrictive to
        # couple the boundary multipliers to the evolution of the primal and dual
        # variables - at least I have some empirical evidence that this can hinder
        # convergence. Might be useful to match IPOPT behaviour here.
        return (step_size * state.step**ω).ω, state.result


# TODO: why don't we check complementarity for the other dual variables? This does not
# make sense - unless we expect poor convergence in these? Perhaps due to the linear
# dependencies of the constraints? Or is it because IPOPT is built with equality
# constraints in mind, as they state in the paper and thesis? For equality constraints,
# complementarity does not need to be checked.
# TODO: I don't support the automatic scaling of the norms yet (and am not sure if I
# should). In IPOPT, this is factor is at least 100, per the documentation and the paper


# TODO: barrier term update. This can be a filter cond at the end of the step, that
# checks for the following things:
# - did convergence occur for the given barrier parameter? -> this is true if terminate
# evaluates to True, so that can be the entry point to the function
# - if convergence occured, then check if the barrier parameter is already down to its
# minimum value. If this is True also, then we terminate, otherwise terminate is
# overwritten. In the latter case, we also need to somehow make the search and descent
# aware that this is the case, to do the following:
# - reset the filter with an extra method on the search
# - update the barrier parameter
# - update the descent so that it is aware of the barrier parameter. It is quite
# probably much cleaner to write this into the function info from inside the solver!!
# (Doesn't it make more sense to write the barrier term into the function info?)
# Alternatively - and not sure what that means for compile time - we could make the
# barrier term an attribute of the solver, and do several solves instead...


def _termination(
    iterate: PyTree,  # pyright: ignore   # TODO API for termination criteria
    f_info: FunctionInfo.EvalGradHessian,
    barrier_parameter,
    tol,
    norm,
):
    assert f_info.constraint_residual is not None
    assert f_info.constraint_jacobians is not None

    y, (equality_dual, inequality_dual), (lb_dual, ub_dual) = iterate

    equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
    dual_term = equality_jacobian.T.mv(equality_dual)  # pyright: ignore  # TODO
    if inequality_jacobian is not None:
        dual_term = (
            dual_term**ω + inequality_jacobian.T.mv(inequality_dual) ** ω  # pyright: ignore
        ).ω
    optimality_error = norm(
        jtu.tree_map(
            lambda a, b, c, d: a + b - c - d, f_info.grad, dual_term, lb_dual, ub_dual
        )
    )  # TODO: compare to IPOPT definition (signs of boundary multipliers)

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

    errata = (optimality_error, constraint_norm)

    lower, upper = f_info.bounds  # pyright: ignore (bounds not None)  # TODO
    lower_diff = (y**ω - lower**ω).ω
    upper_diff = (upper**ω - y**ω).ω
    lower_error = (lower_diff**ω * lb_dual**ω - barrier_parameter).ω
    upper_error = (upper_diff**ω * ub_dual**ω - barrier_parameter).ω
    finite_lower = jtu.tree_map(jnp.isfinite, lower)
    finite_upper = jtu.tree_map(jnp.isfinite, upper)
    lower_error = tree_where(finite_lower, lower_error, 0.0)
    upper_error = tree_where(finite_upper, upper_error, 0.0)
    errata += (norm(lower_error), norm(upper_error))

    return jnp.max(jnp.asarray(errata)) < tol


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


# TODO: update the signs of the constraint functions in the Lagrangian. IPOPT adds the
# constraint term, rather than subtracting it. In other solvers, a negative sign is used
# for the constraint term. I prefer this notation, since it is more straightforward for
# inequality constraints - at least to me it makes more sense that we would want to keep
# both the slack variables and the multipliers strictly positive. In the interest of
# modularity and ease of maintainability, enforcing a common convention here is a good
# choice.
# TODO: what happens when no bounds are specified? The barrier parameter is meaningless
# in this case.
# TODO documentation
# TODO:
# - adaptive update of the barrier parameters
# - testing on a real benchmark problem
# - adding support for inequality constraints
# - trajectory optimisation example
# - iterative refinement
# - KKT error minimisation ahead of robust feasibility restoration
# - inertia correction
# - second-order correction
# - merit function as iterate method?
# - what happens to dual variables after feasibility restoration?
# - building infrastructure for benchmarking problems with pytest-benchmark
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
    of `y`.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, FunctionInfo.EvalGradHessian, Any]]
    search: AbstractVar[
        AbstractSearch[Y, FunctionInfo.EvalGradHessian, FunctionInfo.Eval, Any]
    ]
    verbose: AbstractVar[frozenset[str]]

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
        if constraint is None:
            raise ValueError(
                "IPOPTLike requires constraints. For unconstrained problems, try "
                "an unconstrained minimiser, like `optx.BFGS`."
            )
        else:
            evaluated = evaluate_constraint(constraint, y)
            constraint_residual, constraint_bound, constraint_jacobians = evaluated

        if bounds is None:
            bounds = (tree_full_like(y, -jnp.inf), tree_full_like(y, jnp.inf))

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

        # TODO primal dual iterates
        # TODO: this would need to special case what we do when there is no constraint
        # TODO: initial bound multipliers can be defined as mu / distance
        lower, upper = bounds
        lower_bound_dual = (1 / (y**ω - lower**ω)).ω
        upper_bound_dual = (1 / (upper**ω - y**ω)).ω
        lower_bound_dual = tree_where(
            jtu.tree_map(jnp.isfinite, lower), lower_bound_dual, 0.0
        )
        upper_bound_dual = tree_where(
            jtu.tree_map(jnp.isfinite, upper), upper_bound_dual, 0.0
        )
        equality_dual, inequality_dual = constraint_residual
        equality_dual = tree_full_like(equality_dual, 1.0)
        inequality_dual = tree_full_like(inequality_dual, -1.0)
        iterate = (
            y,
            (equality_dual, inequality_dual),
            (lower_bound_dual, upper_bound_dual),
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
        autodiff_mode = options.get("autodiff_mode", "bwd")
        if bounds is None:
            bounds = (tree_full_like(y, -jnp.inf), tree_full_like(y, jnp.inf))

        # TODO names! duals, boundary_multipliers? constraint_multipliers, bound_mult..?
        y_eval, (equality_dual, inequality_dual), boundary_multipliers = state.iterate

        evaluated = evaluate_constraint(constraint, y_eval)
        constraint_residual, constraint_bound, constraint_jacobians = evaluated

        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), y_eval, has_aux=True
        )

        # TODO: with a second-order correction, all of these become proposed step sizes
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval, bounds, constraint_residual),
            state.search_state,
        )

        def accepted(states):
            search_state, descent_state = states

            grad = lin_to_grad(lin_fn, y_eval, autodiff_mode=autodiff_mode)

            # TODO: WIP: Hessians of the Lagrangian
            hessian_, _ = jax.hessian(fn)(y_eval, args)

            def constraint_hessian(_y, _duals, _template):
                # Template should have the same shape as the hessian of fn, used to
                # correct the shape of the constraint Hessian ((1, 2, 2) --> (2, 2))
                equality_dual, inequality_dual = _duals
                equality_hess, inequality_hess = jax.hessian(constraint)(_y)  # pyright: ignore

                summed_hess = _template
                if equality_dual is not None:
                    if equality_dual.size == 1:
                        weighted_contribution = (equality_dual * equality_hess**ω).ω
                        summed_hess = (summed_hess**ω + weighted_contribution**ω).ω
                    else:
                        for d, h in zip(equality_dual, equality_hess):
                            weighted_contribution = (d * h**ω).ω
                            summed_hess = (summed_hess**ω + weighted_contribution**ω).ω
                if inequality_dual is not None:
                    if inequality_dual.size == 1:
                        weighted_contribution = (inequality_dual * inequality_hess**ω).ω
                        summed_hess = (summed_hess**ω + weighted_contribution**ω).ω
                    else:
                        for d, h in zip(inequality_dual, inequality_hess):
                            weighted_contribution = (d * h**ω).ω
                            summed_hess = (summed_hess**ω + weighted_contribution**ω).ω

                def match_leaf_shape(x, example):  # (1, 2, 2) --> (2, 2)
                    # TODO perhaps rename squeeze if needed (and use squeeze)
                    if x.shape == example.shape:
                        return x
                    else:
                        return jnp.reshape(x, example.shape)

                summed_hess = jtu.tree_map(match_leaf_shape, summed_hess, _template)

                return summed_hess

            dual_ = (equality_dual, inequality_dual)
            template = tree_full_like(hessian_, 0.0)
            constraint_hessian_ = constraint_hessian(y_eval, dual_, template)

            # TODO: too large values of the constraint hessian prevent convergence. Why?
            lagrangian_hessian_ = (hessian_**ω + constraint_hessian_**ω).ω
            lagrangian_hessian_ = lx.PyTreeLinearOperator(
                # TODO: start slowly!
                lagrangian_hessian_,
                jax.eval_shape(lambda: y_eval),
                lx.positive_semidefinite_tag,  # TODO Not technically correct!
            )
            f_eval_info_ = FunctionInfo.EvalGradHessian(
                f_eval,
                grad,
                lagrangian_hessian_,
                y_eval,
                bounds,
                constraint_residual,
                constraint_bound,
                constraint_jacobians,  # pyright: ignore
            )

            # TODO: something going on here with the permitted types, fixing this is
            # punted until we make a decision on whether to unify termination criteria
            # with a common interface
            # TODO: prototyping to get a descent over a pair (primal, duals...)!!!
            iterate = (y_eval, (equality_dual, inequality_dual), boundary_multipliers)
            terminate = _termination(
                iterate,
                f_eval_info_,  # pyright: ignore
                0.01,  # This is supposed to be the barrier parameter
                self.atol,
                self.norm,  # pyright: ignore
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step

            descent_state = self.descent.query(
                iterate,
                f_eval_info_,  # pyright: ignore
                descent_state,
            )

            return (
                y_eval,  # TODO: return iterate? - can reconstruct it from descent state
                f_eval_info_,
                aux_eval,
                search_state,
                descent_state,
                terminate,
            )

        def rejected(states):
            search_state, descent_state = states

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
            )

        # Branch for normal acceptance / rejection
        y, f_info, aux, search_state, descent_state, terminate = filter_cond(
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
                (verbose_y, "y", y_eval),
                (verbose_y, "y on the last accepted step", y),
            )

        descent_steps, descent_result = self.descent.step(step_size, descent_state)
        iterate_eval = (state.iterate**ω + descent_steps**ω).ω
        y_descent, *_ = descent_steps  # TODO: for use in regular_update

        requires_restoration = (
            search_result == RESULTS.feasibility_restoration_required
        ) | (descent_result == RESULTS.feasibility_restoration_required)

        def restore(args):
            del args
            # TODO: make attribute and update the penalty parameter for the feasibility
            # restoration problem based on the barrier parameter.
            boundary_map = ClosestFeasiblePoint(1e-6, BFGS(rtol=1e-3, atol=1e-6))
            recovered_y, restoration_result = boundary_map(y_eval, constraint, bounds)
            # TODO: Allow feasibility restoration to raise a certificate of
            # infeasibility and error out.

            # Re-initialise the search
            f_info_struct = eqx.filter_eval_shape(lambda: state.f_info)
            new_search_state = self.search.init(recovered_y, f_info_struct)
            # TODO: inelegant to use state.iterate here?
            new_descent_state = self.descent.init(state.iterate, f_info_struct)
            return recovered_y, new_search_state, restoration_result, new_descent_state

        def regular_update(args):
            search_result, search_state, descent_result, descent_state = args
            result = RESULTS.where(
                search_result == RESULTS.successful, descent_result, search_result
            )
            y_eval = (y**ω + y_descent**ω).ω
            return y_eval, search_state, result, descent_state

        args = (search_result, search_state, descent_result, descent_state)
        y_eval, search_state, result, descent_state = filter_cond(
            requires_restoration, restore, regular_update, args
        )

        # I currently update the boundary multipliers and y_eval separately
        _, new_dual, new_bounds_iterate_thingy = iterate_eval  # TODO: streamline this
        new_iterate = (y_eval, new_dual, new_bounds_iterate_thingy)
        state = _IPOPTLikeState(
            first_step=jnp.array(False),
            iterate=new_iterate,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
        )
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
    descent: IPOPTLikeDescent
    search: FilteredLineSearch
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = IPOPTLikeDescent()
        self.search = FilteredLineSearch()
        self.verbose = verbose


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
"""
