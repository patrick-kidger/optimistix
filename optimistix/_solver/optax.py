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
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._minimise import AbstractMinimiser
from .._solution import RESULTS


_OptaxClass: TypeAlias = Any
_OptState: TypeAlias = tuple[Int[Array, ""], Any]


class OptaxMinimiser(AbstractMinimiser[_OptState, Y, Aux]):
    optax_cls: _OptaxClass
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    max_steps: int

    def __init__(self, optax_cls, *args, max_steps, **kwargs):
        self.optax_cls = optax_cls
        self.args = args
        self.kwargs = kwargs
        self.max_steps = max_steps

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _OptState:
        del fn, args, options, f_struct, aux_struct
        step_index = jnp.array(0)
        optim = self.optax_cls(*self.args, **self.kwargs)
        opt_state = optim.init(y)
        return step_index, opt_state

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _OptState,
        tags: frozenset[object],
    ) -> tuple[Y, _OptState, Aux]:
        del options

        @eqx.filter_grad
        def compute_grads(_y):
            value = fn(_y, args)
            value, aux = value
            return value, aux

        grads, aux = compute_grads(y)
        step_index, opt_state = state  # pyright: ignore
        optim = self.optax_cls(*self.args, **self.kwargs)
        updates, new_opt_state = optim.update(grads, opt_state)
        new_y = eqx.apply_updates(y, updates)
        new_state = (step_index + 1, new_opt_state)
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _OptState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, y, args, options
        step_index, _ = state
        return jnp.array(step_index > self.max_steps), RESULTS.successful

    def buffers(self, state: _OptState) -> tuple[()]:
        return ()
