from typing import Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import (
    EqualityOut,
    InequalityOut,
    Y,
)
from .._misc import (
    feasible_step_length,
    filter_cond,
    tree_full_like,
    tree_where,
)
from .._search import (
    AbstractDescent,
    FunctionInfo,
    Iterate,
)
from .._solution import RESULTS


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


class _XDYcYdDescentState(
    eqx.Module, Generic[Y, EqualityOut, InequalityOut], strict=True
):
    first_step: Bool[Array, ""]
    step: PyTree
    result: RESULTS


# TODO: This descent implements the XDYcYd descent, as implemented in HiOp. It currently
# requires that bounds be specified as inequality constraints.
# TODO: this thing NEEDS a new name, this name is maximally non-descriptive. (It stands
# for X - D - Yc - Yd, where X is the primal variable, D are the slack variables, Yc are
# the multipliers for the equality constraints and Yd are the multipliers for the
# inequality constraints.
class XDYcYdDescent(
    AbstractDescent[Y, FunctionInfo.EvalGradHessian, _XDYcYdDescentState],
    strict=True,
):
    """TODO: add a description here."""

    linear_solver: lx.AbstractLinearSolver = lx.SVD()

    def init(  # pyright: ignore (figuring out what types a descent should take)
        self, iterate, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _XDYcYdDescentState:
        del f_info_struct
        # TODO: error management here
        return _XDYcYdDescentState(jnp.array(True), iterate, RESULTS.successful)

    def query(  # pyright: ignore  (figuring out what types a descent should take)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _XDYcYdDescentState,
    ) -> _XDYcYdDescentState:
        # Handle initialisation of dual variables where required
        def least_squares_duals(iterate__f_info):
            iterate, f_info = iterate__f_info
            # TODO: no support for no constraints yet! And either constraint must be
            # allowed to be None, but we don't handle this case here
            y, duals, boundary_multipliers = iterate

            def make_kkt(jacobians, input_structure):
                hjac, gjac = jacobians

                def kkt(inputs):
                    aux_step, (eq_dual_step, ineq_dual_step) = inputs
                    r1 = aux_step + hjac.T.mv(eq_dual_step) + gjac.T.mv(ineq_dual_step)
                    r2 = hjac.mv(aux_step)
                    r3 = gjac.mv(aux_step)
                    return r1, (r2, r3)

                return lx.FunctionLinearOperator(kkt, input_structure)

            input_structure = jax.eval_shape(lambda: (y, duals))
            kkt_operator = make_kkt(f_info.constraint_jacobians, input_structure)
            vector = (-f_info.grad, tree_full_like(duals, 0.0))

            out = lx.linear_solve(kkt_operator, vector, self.linear_solver)
            _, new_duals = out.value

            # TODO: make the cutoff a descent attribute
            reasonable_duals = jtu.tree_map(lambda x: jnp.abs(x) < 1e3, new_duals)
            new_duals = tree_where(reasonable_duals, new_duals, 0.0)

            new_iterate = (y, new_duals, boundary_multipliers)

            return new_iterate

        def keep_duals(iterate__f_info):
            iterate, _ = iterate__f_info
            return iterate

        args = (iterate, f_info)
        iterate = filter_cond(state.first_step, least_squares_duals, keep_duals, args)

        # TODO: IPOPTLike defaults no bounds to infinite bounds, but we ignore bounds
        # here. (This anyway needs to be rectified.)
        y, (equality_dual, inequality_dual), boundary_multipliers = iterate

        # TODO: test driving the iterate here
        _, slack = f_info.constraint_residual  # pyright: ignore
        iterate_ = Iterate.AllAtOnce(
            y=y,
            bounds=f_info.bounds,
            slack=slack,
            equality_dual=equality_dual,
            inequality_dual=inequality_dual,
            boundary_multipliers=boundary_multipliers,
        )

        # TODO: special case for infinite bounds? Right now not doing that
        equality_residual, inequality_residual = f_info.constraint_residual  # pyright: ignore
        barrier_hessian = jtu.tree_map(
            lambda s, d: d / (s + 1e-6), iterate_.slack, iterate_.inequality_dual
        )
        barrier_hessian = lx.DiagonalLinearOperator(barrier_hessian)

        # TODO: adapt for PyTrees
        def make_kkt_operator(w, z, jacs):
            hj, gj = jacs

            def kkt(inputs):
                y, slacks, (equality_duals, inequality_duals) = inputs
                y_step = w.mv(y) + hj.T.mv(equality_duals) + gj.T.mv(inequality_duals)
                s_step = -z.mv(slacks) + inequality_duals  # TODO is this correct?
                hl_step = hj.mv(y)
                gl_step = gj.mv(y) + slacks
                return y_step, s_step, (hl_step, gl_step)

            return kkt

        kkt_operator_ = make_kkt_operator(
            f_info.hessian, barrier_hessian, f_info.constraint_jacobians
        )
        input_structure = jax.eval_shape(
            lambda: (
                iterate_.y,
                iterate_.slack,
                (iterate_.equality_dual, iterate_.inequality_dual),
            )
        )
        kkt_operator = lx.FunctionLinearOperator(kkt_operator_, input_structure)

        hj, gj = f_info.constraint_jacobians  # pyright: ignore
        barrier = 0.01  # TODO

        eq_dual = iterate_.equality_dual
        ineq_dual = iterate_.inequality_dual
        upper_rhs = f_info.grad + hj.T.mv(eq_dual) + gj.T.mv(ineq_dual)  # pyright: ignore
        vector = (
            -upper_rhs,  # pyright: ignore
            # TODO: hard-coded barrier parameter here, and hard-coded epsilon
            -iterate_.inequality_dual + barrier * 1 / (iterate_.slack + 1e-6),  # pyright: ignore
            (-equality_residual, tree_full_like(inequality_residual, 0.0)),
            # TODO: no explicit slack variable (0.0) - this is recommended in N & W (!)
            # (At least that is my interpretation of Chapter 19.)
            # TODO: residuals - also for constraint function - in f_info
            # TODO: no boundary multipliers yet
        )
        # TODO: what is meant by the last bit? I don't have explicit slack variables
        # yet, I "initialise" the slack variables as the residuals of the constraints
        # at each iteration. I think in practice these might deviate!!

        out = lx.linear_solve(kkt_operator, vector, self.linear_solver)

        # TODO: truncate to feasible step size

        # Output while I figure out the other steps
        y_step, slack_steps, dual_steps = out.value  # TODO: throw the slack steps away

        dual_steps = (dual_steps**ω).ω  # Flip sign to see if this is enough to get it
        # to work. The system is now equivalent to the one given in 19.3 of NW on p. 569
        slack_steps = (
            -(slack_steps**ω)
        ).ω  # Only relevant to constrain step length TODO
        max_slack_step_size = feasible_step_length(
            inequality_residual,
            tree_full_like(inequality_residual, 0.0),
            slack_steps,
            offset=barrier,
        )
        _, inequality_dual_steps = dual_steps
        max_dual_step_size = feasible_step_length(
            inequality_dual,
            tree_full_like(inequality_dual, 0.0),
            inequality_dual_steps,
            offset=barrier,
        )
        max_step_size = jnp.min(jnp.array([max_slack_step_size, max_dual_step_size]))
        # TODO boundary multiplier steps

        # TODO: boundary multipliers currently do nothing / but not so fake anymore
        fake_iterate_step = (y_step, dual_steps, boundary_multipliers)
        fake_iterate_step = (max_step_size * fake_iterate_step**ω).ω

        return _XDYcYdDescentState(
            jnp.array(False),
            fake_iterate_step,
            RESULTS.promote(out.result),
        )

    def step(self, step_size: Scalar, state: _XDYcYdDescentState) -> tuple[Y, RESULTS]:
        # TODO Note that I *am* currently scaling the dual variables for the bounds too
        # Fixing this would require unpacking the tuple and packing it up again
        # When very large values for the bounds are used, it does seem restrictive to
        # couple the boundary multipliers to the evolution of the primal and dual
        # variables - at least I have some empirical evidence that this can hinder
        # convergence. Might be useful to match IPOPT behaviour here.
        return (step_size * state.step**ω).ω, state.result
