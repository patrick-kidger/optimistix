from typing import Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from .._custom_types import (
    EqualityOut,
    InequalityOut,
    Y,
)
from .._misc import (
    tree_full_like,
    tree_where,
)
from .._search import (
    AbstractDescent,
    FunctionInfo,
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
        return _XDYcYdDescentState(iterate, RESULTS.successful)

    def query(  # pyright: ignore  (figuring out what types a descent should take)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _XDYcYdDescentState,
    ) -> _XDYcYdDescentState:
        # TODO: IPOPTLike defaults no bounds to infinite bounds, but we ignore bounds
        # here. (This anyway needs to be rectified.)
        y, (equality_dual, inequality_dual), boundary_multipliers = iterate

        # TODO: special case for infinite bounds? Right now not doing that
        equality_residual, inequality_residual = f_info.constraint_residual  # pyright: ignore
        barrier_hessian = jtu.tree_map(
            lambda s, d: d / (s + 1e-6), inequality_residual, inequality_dual
        )
        barrier_hessian = lx.DiagonalLinearOperator(barrier_hessian)
        # jax.debug.print("Barrier Hessian in descent.query: \n{}", barrier_hessian)

        # TODO: adapt for PyTrees
        def make_kkt_operator(w, z, jacs):
            hj, gj = jacs

            def kkt(inputs):
                y, slacks, (equality_duals, inequality_duals) = inputs
                y_step = w.mv(y) + hj.T.mv(equality_duals) + gj.T.mv(inequality_duals)
                s_step = z.mv(slacks) - inequality_duals
                hl_step = hj.mv(y)
                gl_step = gj.mv(y) - slacks
                return y_step, s_step, (hl_step, gl_step)

            return kkt

        kkt_operator_ = make_kkt_operator(
            f_info.hessian, barrier_hessian, f_info.constraint_jacobians
        )
        input_structure = jax.eval_shape(
            lambda: (y, inequality_residual, (equality_dual, inequality_dual))
        )
        kkt_operator = lx.FunctionLinearOperator(kkt_operator_, input_structure)
        # jax.debug.print("KKT matrix: \n{}", kkt_operator.as_matrix())

        hj, gj = f_info.constraint_jacobians  # pyright: ignore
        vector = (
            -f_info.grad
            + hj.T.mv(equality_dual)  # pyright: ignore
            + gj.T.mv(inequality_dual),  # pyright: ignore
            # TODO: hard-coded barrier parameter here, and hard-coded epsilon
            -inequality_dual + 0.01 * 1 / (inequality_residual + 1e-6),
            (-equality_residual, 0.0),  # TODO: no explicit slack variable
        )
        # TODO: what is meant by the last bit? I don't have explicit slack variables
        # yet, I "initialise" the slack variables as the residuals of the constraints
        # at each iteration. I think in practice these might deviate!!

        # jax.debug.print("vector: \n{}", vector)

        out = lx.linear_solve(kkt_operator, vector, self.linear_solver)
        # jax.debug.print("out.value: \n{}", out.value)

        # TODO: truncate to feasible step size

        # Output while I figure out the other steps
        y_step, _, dual_steps = out.value  # TODO: throw the slack steps away
        # TODO: boundary multipliers currently do nothing
        fake_iterate_step = (y_step, dual_steps, boundary_multipliers)
        # result = RESULTS.successful

        return _XDYcYdDescentState(
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
