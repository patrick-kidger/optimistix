# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, cast, Optional

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._line_search import AbstractDescent
from .._misc import sum_squares, tree_inner_prod, tree_where
from .._solution import RESULTS


def _sum_leaves(tree: PyTree[Array]):
    return jtu.tree_reduce(lambda x, y: x + y, tree)


def _gradient_step(
    vector: PyTree[Array], vector_prev: PyTree[Array], diff_prev: PyTree[Array]
) -> Scalar:
    return jnp.array(0.0)


def fletcher_reeves(
    vector: PyTree[Array], vector_prev: PyTree[Array], diff_prev: PyTree[Array]
) -> Scalar:
    numerator = sum_squares(vector)
    denominator = sum_squares(vector_prev)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def polak_ribiere(
    vector: PyTree[Array], vector_prev: PyTree[Array], diff_prev: PyTree[Array]
) -> Scalar:
    numerator = tree_inner_prod(vector, (vector**ω - vector_prev**ω).ω)
    denominator = sum_squares(vector_prev)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    beta = cast(
        Scalar, jnp.where(pred, jnp.clip(numerator / safe_denom, a_min=0), jnp.inf)
    )
    return beta


def hestenes_stiefel(
    vector: PyTree[Array], vector_prev: PyTree[Array], diff_prev: PyTree[Array]
) -> Scalar:
    grad_diff = (vector**ω - vector_prev**ω).ω
    numerator = tree_inner_prod(vector, grad_diff)
    denominator = tree_inner_prod(diff_prev, grad_diff)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def dai_yuan(
    vector: PyTree[Array], vector_prev: PyTree[Array], diff_prev: PyTree[Array]
) -> Scalar:
    numerator = sum_squares(vector)
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
        fn: Fn[Y, Out, Aux],
        y: Y,
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: Any,
        options: dict[str, Any],
    ):
        return NonlinearCGState(vector, vector, vector, jnp.array(0))

    def update_state(
        self,
        descent_state: NonlinearCGState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
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
    ) -> tuple[PyTree[Array], RESULTS]:
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
        return (delta * diff**ω).ω, RESULTS.successful

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: NonlinearCGState,
        args: Any,
        options: dict[str, Any],
    ):
        assert False
