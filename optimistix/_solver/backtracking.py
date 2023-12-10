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

from typing import cast, Union
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Scalar, ScalarLike

from .._custom_types import Y
from .._misc import (
    tree_dot,
)
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


class _BacktrackingState(eqx.Module):
    step_size: Scalar


_FnInfo: TypeAlias = Union[
    FunctionInfo.EvalGrad,
    FunctionInfo.EvalGradHessian,
    FunctionInfo.EvalGradHessianInv,
    FunctionInfo.ResidualJac,
]
_FnEvalInfo: TypeAlias = FunctionInfo


class BacktrackingArmijo(AbstractSearch[Y, _FnInfo, _FnEvalInfo, _BacktrackingState]):
    """Perform a backtracking Armijo line search."""

    decrease_factor: ScalarLike = 0.5
    slope: ScalarLike = 0.1
    step_init: ScalarLike = 1.0

    def __post_init__(self):
        self.decrease_factor = eqx.error_if(
            self.decrease_factor,
            (self.decrease_factor <= 0)  # pyright: ignore
            | (self.decrease_factor >= 1),  # pyright: ignore
            "`BacktrackingArmoji(decrease_factor=...)` must be between 0 and 1.",
        )
        self.slope = eqx.error_if(
            self.slope,
            (self.slope <= 0) | (self.slope >= 1),  # pyright: ignore
            "`BacktrackingArmoji(slope=...)` must be between 0 and 1.",
        )
        self.step_init = eqx.error_if(
            self.step_init,
            self.step_init <= 0,  # pyright: ignore
            "`BacktrackingArmoji(step_init=...)` must be strictly greater than 0.",
        )

    def init(self, y: Y, f_info_struct: _FnInfo) -> _BacktrackingState:
        del f_info_struct
        return _BacktrackingState(step_size=jnp.array(self.step_init))

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: _BacktrackingState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _BacktrackingState]:
        if isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
                FunctionInfo.ResidualJac,
            ),
        ):
            grad = f_info.grad
        else:
            raise ValueError(
                "Cannot use `BacktrackingArmijo` with this solver. This is because "
                "`BacktrackingArmijo` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )

        y_diff = (y_eval**ω - y**ω).ω
        predicted_reduction = tree_dot(grad, y_diff)
        # Terminate when the Armijo condition is satisfied. That is, `fn(y_eval)`
        # must do better than its linear approximation:
        # `fn(y_eval) < fn(y) + grad•y_diff`
        f_min = f_info.as_min()
        f_min_eval = f_eval_info.as_min()
        f_min_diff = f_min_eval - f_min  # This number is probably negative
        satisfies_armijo = f_min_diff <= self.slope * predicted_reduction
        has_reduction = predicted_reduction <= 0

        accept = first_step | (satisfies_armijo & has_reduction)
        step_size = jnp.where(
            accept, self.step_init, self.decrease_factor * state.step_size
        )
        step_size = cast(Scalar, step_size)
        return (
            step_size,
            accept,
            RESULTS.successful,
            _BacktrackingState(step_size=step_size),
        )


BacktrackingArmijo.__init__.__doc__ = """**Arguments:**

- `decrease_factor`: The rate at which to backtrack, i.e.
    `next_stepsize = decrease_factor * current_stepsize`. Must be between 0 and 1.
- `slope`: The slope of of the linear approximation to
    `f` that the backtracking algorithm must exceed to terminate. Larger
    means stricter termination criteria. Must be between 0 and 1.
- `step_init`: The first `step_size` the backtracking algorithm will
    try. Must be greater than 0.
"""
