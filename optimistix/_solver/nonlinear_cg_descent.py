from typing import Any, Callable, Optional

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from .._iterate import AbstractIterativeProblem
from .._line_search import AbstractDescent
from .._misc import tree_inner_prod, tree_where, two_norm
from .._solution import RESULTS


def _sum_leaves(tree):
    return jtu.tree_reduce(lambda x, y: x + y, tree)


def _gradient_step(vector, vector_prev, diff_prev):
    return jnp.array(0.0)


def fletcher_reeves(vector, vector_prev, diff_prev):
    numerator = two_norm(vector) ** 2
    denominator = two_norm(vector_prev) ** 2
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def polak_ribiere(vector, vector_prev, diff_prev):
    numerator = tree_inner_prod(vector, (vector**ω - vector_prev**ω).ω)
    denominator = two_norm(vector_prev) ** 2
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    beta = jnp.where(pred, jnp.clip(numerator / safe_denom, a_min=0), jnp.inf)
    return beta


def hestenes_stiefel(vector, vector_prev, diff_prev):
    grad_diff = (vector**ω - vector_prev**ω).ω
    numerator = tree_inner_prod(vector, grad_diff)
    denominator = tree_inner_prod(diff_prev, grad_diff)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def dai_yuan(vector, vector_prev, diff_prev):
    numerator = two_norm(vector) ** 2
    denominator = tree_inner_prod(diff_prev, (vector**ω - vector_prev**ω).ω)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


class NonlinearCGState(eqx.Module):
    vector: PyTree[Array]
    vector_prev: PyTree[Array]
    diff_prev: PyTree[Array]
    step: PyTree[Array]


class NonlinearCGDescent(AbstractDescent[NonlinearCGState]):
    method: Callable

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return NonlinearCGState(vector, vector, vector, jnp.array(0))

    def update_state(
        self,
        descent_state: NonlinearCGState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: Optional[dict[str, Any]] = None,
    ):
        # not sure of a better way to do this at the moment
        beta = lax.cond(
            descent_state.step > 1,
            self.method,
            _gradient_step,
            descent_state.vector,
            descent_state.vector_prev,
            descent_state.diff_prev,
        )
        diff_prev = (-ω(descent_state.vector) + beta * ω(descent_state.diff_prev)).ω
        return NonlinearCGState(
            vector, descent_state.vector, diff_prev, descent_state.step + 1
        )

    def __call__(
        self,
        delta: Scalar,
        descent_state: NonlinearCGState,
        args: Any,
        options: dict[str, Any],
    ):
        beta = lax.cond(
            descent_state.step > 1,
            self.method,
            _gradient_step,
            descent_state.vector,
            descent_state.vector_prev,
            descent_state.diff_prev,
        )
        negative_gradient = (-descent_state.vector**ω).ω
        nonlinear_cg_direction = (
            negative_gradient**ω + beta * descent_state.diff_prev**ω
        ).ω
        is_descent_direction = (
            tree_inner_prod(descent_state.vector, nonlinear_cg_direction) < 0
        )
        diff = tree_where(
            is_descent_direction, nonlinear_cg_direction, negative_gradient
        )
        return (delta * diff**ω).ω, jnp.array(RESULTS.successful)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: NonlinearCGState,
        args: Any,
        options: dict[str, Any],
    ):
        assert False
