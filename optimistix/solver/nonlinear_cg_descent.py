from typing import Any, Callable, Optional

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from ..custom_types import Scalar
from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractDescent
from ..linear_operator import AbstractLinearOperator
from ..misc import tree_inner_prod, two_norm
from ..solution import RESULTS


def _sum_leaves(tree):
    return jtu.tree_reduce(lambda x, y: x + y, tree)


def _gradient_step(vector, vector_prev, diff_prev):
    return jnp.array(0.0)


def fletcher_reeves(vector, vector_prev, diff_prev):
    denominator = two_norm(vector_prev) ** 2
    return two_norm(vector) ** 2 / denominator


def polak_ribiere(vector, vector_prev, diff_prev):
    numerator = tree_inner_prod(vector, (vector**ω - vector_prev**ω).ω)
    denominator = two_norm(vector_prev) ** 2
    beta = numerator / denominator
    return jnp.clip(beta, a_min=0, a_max=100)


def hestenes_stiefel(vector, vector_prev, diff_prev):
    grad_diff = (vector**ω - vector_prev**ω).ω
    numerator = tree_inner_prod(vector, grad_diff)
    denominator = tree_inner_prod(diff_prev, grad_diff)
    beta = numerator / denominator
    return jnp.clip(beta, a_min=0)


def dai_yuan(vector, vector_prev, diff_prev):
    denominator = tree_inner_prod(diff_prev, (vector**ω - vector_prev**ω).ω)
    return two_norm(vector) ** 2 / denominator


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
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return NonlinearCGState(vector, vector, vector, jnp.array(0))

    def update_state(
        self,
        descent_state: NonlinearCGState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        options: Optional[dict[str, Any]] = None,
    ):
        # not sure of a better way to do this at the moment
        beta = lax.cond(
            descent_state.step > 3,
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
            descent_state.step > 3,
            self.method,
            _gradient_step,
            descent_state.vector,
            descent_state.vector_prev,
            descent_state.diff_prev,
        )
        return (
            delta * (-ω(descent_state.vector) + beta * ω(descent_state.diff_prev))
        ).ω, jnp.array(RESULTS.successful)
