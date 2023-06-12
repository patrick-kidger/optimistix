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

from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Fn, LineSearchAux
from .._line_search import AbstractLineSearch
from .._solution import RESULTS


class LearningRate(AbstractLineSearch[Bool[Array, ""]]):
    learning_rate: Scalar

    def first_init(
        self,
        vector: PyTree[Array],
        operator: lx.AbstractLinearOperator,
        options: dict[str, Any],
    ) -> Scalar:
        return self.learning_rate

    def init(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> Bool[Array, ""]:
        # Needs to run once, so set the state to be whether or not
        # it has ran yet.
        return jnp.array(False)

    def step(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: Bool[Array, ""],
        tags: frozenset[object],
    ) -> tuple[Scalar, Bool[Array, ""], LineSearchAux]:
        (f_val, (_, diff, aux, result, _)) = fn(y, args)
        return (
            y,
            jnp.array(True),
            (f_val, diff, aux, result, self.learning_rate),
        )

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: Bool[Array, ""],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state, RESULTS.successful

    def buffers(self, state: Bool[Array, ""]) -> tuple[()]:
        return ()
