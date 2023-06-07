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

from typing import Any, Sequence, TYPE_CHECKING, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._minimise import AbstractMinimiser
from .._misc import tree_where


if TYPE_CHECKING:
    _Node = Any
else:
    _Node = eqxi.doc_repr(Any, "Node")


def _auxmented(fn, x, args):
    z, original_aux = fn(x, args)
    aux = (z, original_aux)
    return z, aux


class RunningMinMinimiser(AbstractMinimiser):
    minimiser: AbstractMinimiser

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct] | None,
        tags: frozenset[object],
    ):
        auxmented = eqx.Partial(_auxmented, fn)
        return self.minimiser.init(
            auxmented, y, args, options, f_struct, aux_struct, tags
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: PyTree,
        tags: frozenset[object],
    ):
        # Keep track of the running min and output this. keep track of the running
        # iterate through state instead and manage this by augmenting the auxiliary
        # output.
        minimiser_state, state_y, f_min = state
        auxmented = eqx.Partial(_auxmented, fn)
        y_new, minimiser_state_new, (f_val, aux) = self.minimiser.step(
            auxmented, state_y, args, options, minimiser_state, tags
        )
        new_best = f_val < f_min
        y_min = tree_where(new_best, y_new, y)
        f_min_new = jnp.where(new_best, f_val, f_min)
        new_state = (minimiser_state_new, y_new, f_min_new)
        return y_min, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: PyTree,
        tags: frozenset[object],
    ):
        minimiser_state, state_y, _ = state
        return self.minimiser.terminate(
            fn, state_y, args, options, minimiser_state, tags
        )

    def buffers(self, state: PyTree) -> Union[_Node, Sequence[_Node]]:
        minimiser_state, *_ = state
        return self.minimiser.buffers(minimiser_state)
