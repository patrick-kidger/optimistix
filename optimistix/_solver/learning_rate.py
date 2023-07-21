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
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar, ScalarLike

from .._adjoint import AbstractAdjoint
from .._custom_types import AbstractLineSearchState, Aux, Fn, Y
from .._line_search import AbstractDescent, AbstractLineSearch
from .._solution import RESULTS, Solution


class _LearningRateState(AbstractLineSearchState):
    next_init: Scalar


class LearningRate(AbstractLineSearch[Y, Aux, _LearningRateState]):
    """Compute `y_new` from `y`, by taking a step of the fixed size `learning_rate`.

    This requires the following `options`:

    - `aux`: The auxiliary output of the function at the point `y`.
    """

    descent: AbstractDescent[Y]
    learning_rate: ScalarLike = eqx.field(converter=jnp.asarray)

    def solve(
        self,
        fn: Fn[Y, Scalar, Aux],
        y0: PyTree[Array],
        args: PyTree,
        options: dict[str, Any],
        *,
        max_steps: Optional[int],
        adjoint: AbstractAdjoint,
        throw: bool,
        tags: frozenset[object],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> Solution[Y, Aux, _LearningRateState]:
        del max_steps, adjoint, throw, tags, f_struct, aux_struct
        diff, _ = self.descent(cast(Array, self.learning_rate), args, options)
        value = (y0**ω + diff**ω).ω
        return Solution(
            value=value,
            result=RESULTS.successful,
            aux=options["aux"],
            stats={},
            state=_LearningRateState(self.learning_rate),
        )


LearningRate.__init__.__doc__ = """**Arguments:**

- `descent`: An [`optimistix.AbstractDescent`][] object, describing how to map from
    step-size (a scalar) to update (in y-space).
- `learning_rate`: The fixed step-size used at each step.
"""
