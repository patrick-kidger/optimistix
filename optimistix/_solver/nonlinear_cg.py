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

from collections.abc import Callable
from typing import Any, cast

import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from .._custom_types import Aux, Y
from .._line_search import AbstractDescent, AbstractLineSearch
from .._misc import (
    max_norm,
    sum_squares,
    tree_dot,
    tree_where,
)
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gradient_methods import AbstractGradientDescent


def _gradient_step(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    return jnp.array(0.0)


def polak_ribiere(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    """The Polak--Ribière formula for β."""
    del diff_prev
    numerator = tree_dot(vector, (vector**ω - vector_prev**ω).ω)
    denominator = sum_squares(vector_prev)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    out = jnp.where(pred, jnp.clip(numerator / safe_denom, a_min=0), jnp.inf)
    return cast(Scalar, out)


def fletcher_reeves(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    """The Fletcher--Reeves formula for β."""
    del diff_prev
    numerator = sum_squares(vector)
    denominator = sum_squares(vector_prev)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def hestenes_stiefel(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    """The Hestenes--Stiefel formula for β."""
    grad_diff = (vector**ω - vector_prev**ω).ω
    numerator = tree_dot(vector, grad_diff)
    denominator = -tree_dot(diff_prev, grad_diff)
    pred = jnp.abs(denominator) > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def dai_yuan(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    """The Dai--Yuan formula for β."""
    numerator = sum_squares(vector)
    denominator = -tree_dot(diff_prev, (vector**ω - vector_prev**ω).ω)
    pred = jnp.abs(denominator) > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


class NonlinearCGDescent(AbstractDescent[Y]):
    """The nonlinear conjugate gradient step.

    This requires the following `options`:

    - `vector`: The gradient of the objective function to minimise.
    - `vector_prev`: The gradient of the objective function to minimise at the
        previous step.
    - `diff_prev`: The `diff` of the previous step. (That is to say, the output of the
        previous call to `NonlinearCGDescent`.)
    """

    method: Callable[[Y, Y, Y], Scalar]

    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        vector = options["vector"]
        vector_prev = options["vector_prev"]
        diff_prev = options["diff_prev"]
        # TODO(kidger): this check seems like overkill? Can we just check if we're on
        # the first step or not?
        isfinite = jnp.isfinite(max_norm(vector_prev)) & jnp.isfinite(
            max_norm(diff_prev)
        )
        beta = jnp.where(isfinite, self.method(vector, vector_prev, diff_prev), 0)
        negative_gradient = (-(vector**ω)).ω
        nonlinear_cg_direction = (negative_gradient**ω + beta * diff_prev**ω).ω
        diff = tree_where(
            tree_dot(vector, nonlinear_cg_direction) < 0,
            nonlinear_cg_direction,
            negative_gradient,
        )
        return (step_size * diff**ω).ω, RESULTS.successful


NonlinearCGDescent.__init__.__doc__ = """**Arguments:**

- `method`: A callable `method(vector, vector_prev, diff_prev)` describing how to
    calculate the beta parameter of nonlinear CG. Each of these inputs has the meaning
    described above. The "beta parameter" is the sake as can be described as e.g. the
    β_n value
    [on Wikipedia](https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method).
    In practice Optimistix includes four built-in methods:
    [`optimistix.polak_ribiere`][], [`optimistix.fletcher_reeves`][],
    [`optimistix.hestenes_stiefel`][], and [`optimistix.dai_yuan`][].
"""


# TODO(raderj): replace the default line search in all of these with a better
# line search algorithm.


class NonlinearCG(AbstractGradientDescent[Y, Aux]):
    """The nonlinear conjugate gradient method."""

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: AbstractDescent
    line_search: AbstractLineSearch

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Scalar] = max_norm,
        method: Callable[[Y, Y, Y], Scalar] = polak_ribiere,
    ):
        """**Arguments:**

        - `rtol`: Relative tolerance for terminating solve.
        - `atol`: Absolute tolerance for terminating solve.
        - `norm`: The norm used to determine the difference between two iterates in the
            convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
            includes three built-in norms: [`optimistix.max_norm`][],
            [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
        - `method`: The function which computes `beta` in `NonlinearCG`. Defaults to
            `polak_ribiere`. Optimistix includes four built-in methods:
            [`optimistix.polak_ribiere`][], [`optimistix.fletcher_reeves`][],
            [`optimistix.hestenes_stiefel`][], and [`optimistix.dai_yuan`][], but any
            function `(Y, Y, Y) -> Scalar` will work.
        """
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NonlinearCGDescent(method=method)
        self.line_search = BacktrackingArmijo(decrease_factor=0.5, backtrack_slope=0.1)
