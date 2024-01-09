# Copyright 2023 Google LLC #
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

from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Scalar, ScalarLike

from .._custom_types import Y
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


class LearningRate(AbstractSearch[Y, FunctionInfo, FunctionInfo, None], strict=True):
    """Move downhill by taking a step of the fixed size `learning_rate`."""

    learning_rate: ScalarLike = eqx.field(converter=jnp.asarray)

    def init(self, y: Y, f_info_struct: FunctionInfo) -> None:
        return None

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo,
        f_eval_info: FunctionInfo,
        state: None,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, None]:
        del first_step, y, y_eval, f_info, f_eval_info, state
        learning_rate = cast(Array, self.learning_rate)
        return learning_rate, jnp.array(True), RESULTS.successful, None


LearningRate.__init__.__doc__ = """**Arguments:**

- `learning_rate`: The fixed step-size used at each step.
"""
