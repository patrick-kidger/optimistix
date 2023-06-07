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

from typing import Any, cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Aux, Fn
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


class _BisectionState(eqx.Module):
    lower: Scalar
    upper: Scalar
    error: Scalar
    flip: Bool[Array, ""]
    step: Int[Array, ""]


class Bisection(AbstractRootFinder[_BisectionState, Scalar, Scalar, Aux]):
    rtol: float
    atol: float

    def init(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BisectionState:
        del f_struct, aux_struct
        upper = options["upper"]
        lower = options["lower"]
        if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function taking a "
                "scalar input"
            )
        out_struct, _ = jax.eval_shape(fn, y, args)
        if not isinstance(out_struct, jax.ShapeDtypeStruct) or out_struct.shape != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function producing "
                "a scalar output"
            )
        # This computes a range such that `f(0.5 * (a+b))` is
        # the user-passed `lower` on the first step, and the user
        # passed `upper` on the second step. This saves us from
        # compiling `fn` two extra times in the init.
        range = upper - lower
        extended_upper = upper + range
        extended_range = extended_upper - lower
        extended_lower = lower - extended_range
        return _BisectionState(
            lower=extended_lower,
            upper=extended_upper,
            error=jnp.asarray(jnp.inf),
            flip=jnp.array(False),
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _BisectionState, Aux]:
        del y, options
        new_y = state.lower + 0.5 * (state.upper - state.lower)
        error, aux = fn(new_y, args)
        too_large = cast(Bool[Array, ""], state.flip ^ (error < 0))
        too_large = jnp.where(state.step == 0, True, too_large)
        too_large = jnp.where(state.step == 1, False, too_large)
        new_lower = jnp.where(too_large, new_y, state.lower)
        new_upper = jnp.where(too_large, state.upper, new_y)
        flip = jnp.where(state.step == 1, error < 0, state.flip)
        message = "The root is not contained in [lower, upper]"
        step = eqxi.error_if(
            state.step, (state.step == 1) & (state.error * error > 0), message
        )
        new_state = _BisectionState(
            lower=new_lower,
            upper=new_upper,
            error=error,
            flip=flip,
            step=step + 1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        return jnp.abs(state.error) < scale, RESULTS.successful

    def buffers(self, state: _BisectionState) -> tuple[()]:
        return ()
