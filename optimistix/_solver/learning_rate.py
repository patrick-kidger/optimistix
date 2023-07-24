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

from typing import Any, cast, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Scalar, ScalarLike

from .._custom_types import DescentState, NoAuxFn, Out, Y
from .._search import AbstractDescent, AbstractSearch, DerivativeInfo
from .._solution import RESULTS


class LearningRate(AbstractSearch[Y, Out, DescentState]):
    """Move downhill by taking a step of the fixed size `learning_rate`."""

    learning_rate: ScalarLike = eqx.field(converter=jnp.asarray)

    def init(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Scalar],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> DescentState:
        return descent.optim_init(fn, y, args, f_struct)

    def search(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: DescentState,
        deriv_info: DerivativeInfo,
        max_steps: Optional[int],
    ) -> tuple[Y, Bool[Array, ""], RESULTS, DescentState]:
        state = descent.search_init(fn, y, args, f, state, deriv_info)
        learning_rate = cast(Array, self.learning_rate)
        y_diff, result, state = descent.descend(
            learning_rate, fn, y, args, f, state, deriv_info
        )
        return y_diff, jnp.array(True), result, state


LearningRate.__init__.__doc__ = """**Arguments:**

- `learning_rate`: The fixed step-size used at each step.
"""
