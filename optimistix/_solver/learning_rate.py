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

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import AbstractLineSearchState, Aux, Fn, Y
from .._descent import AbstractDescent
from .._minimise import AbstractMinimiser
from .._misc import tree_full_like
from .._solution import RESULTS


class _LearningRateState(AbstractLineSearchState):
    next_init: Scalar
    finished: Bool[Array, ""]
    aux: PyTree


class LearningRate(AbstractMinimiser[_LearningRateState, Y, Aux]):
    descent: AbstractDescent[Y]
    learning_rate: Scalar = eqx.field(converter=jnp.asarray)

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _LearningRateState:
        del fn, y, args, f_struct
        aux = tree_full_like(aux_struct, 0.0)
        return _LearningRateState(
            self.learning_rate, finished=jnp.array(False), aux=aux
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _LearningRateState,
        tags: frozenset[object],
    ) -> tuple[Y, _LearningRateState, Aux]:
        del fn, tags
        diff, _ = self.descent(state.next_init, args, options)
        new_state = _LearningRateState(
            self.learning_rate, jnp.array(True), aux=state.aux
        )
        return (y**ω + diff**ω).ω, new_state, state.aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _LearningRateState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, y, args, options, tags
        return state.finished, RESULTS.successful

    def buffers(self, state: _LearningRateState) -> tuple[()]:
        return ()
