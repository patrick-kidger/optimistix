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
from typing import Any, cast, ClassVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._base_solver import AbstractHasTol
from .._custom_types import Aux, Fn
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


class _BisectionState(eqx.Module):
    lower: Scalar
    upper: Scalar
    flip: Bool[Array, ""]
    step: Int[Array, ""]


class Bisection(
    AbstractRootFinder[Scalar, Scalar, Aux, _BisectionState],
    AbstractHasTol,
):
    """The bisection method of root finding. This may only be used with functions
    `R->R`, i.e. functions with scalar input and scalar output.

    This requires the following `options`:

    - `lower`: The lower bound on the interval which contains the root.
    - `upper`: The upper bound on the interval which contains the root.

    This algorithm works by considering the interval `[lower, upper]`, checking the
    sign of the evaluated function at the midpoint of the interval, and then keeping
    whichever half contains the root. This is then repeated. The iteration stops once
    the interval is sufficiently small.
    """

    rtol: float
    atol: float
    # All norms are the same for scalars.
    norm: ClassVar[Callable[[PyTree], Scalar]] = jnp.abs

    def init(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BisectionState:
        lower = options["lower"]
        upper = options["upper"]
        del options, aux_struct
        if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function producing "
                "a scalar output."
            )
        # `extended_upper` and `extended_lower` form a range such that
        # `f(0.5 * (a+b))` is the user-passed `lower` on the first step,
        # and the user passed `upper` on the second step. This saves us from
        # compiling `fn` two extra times in the init.
        range = upper - lower
        extended_upper = upper + range
        extended_range = extended_upper - lower
        extended_lower = lower - extended_range
        return _BisectionState(
            lower=jnp.asarray(extended_lower, f_struct.dtype),
            upper=jnp.asarray(extended_upper, f_struct.dtype),
            flip=jnp.array(False),
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _BisectionState, Aux]:
        del y, options
        new_y = state.lower + 0.5 * (state.upper - state.lower)
        error, aux = fn(new_y, args)
        too_large = cast(Bool[Array, ""], state.flip ^ (error < 0))
        # On step 0 and 1 we set `too_large` to update `state.lower` and `state.upper`
        # to match the values set by the user instead of the extended range computed in
        # the init.
        too_large = jnp.where(state.step == 0, True, too_large)
        too_large = jnp.where(state.step == 1, False, too_large)
        new_lower = jnp.where(too_large, new_y, state.lower)
        new_upper = jnp.where(too_large, state.upper, new_y)
        flip = jnp.where(state.step < 2, error < 0, state.flip)
        # `step` is passed through to make sure this error check does not get DCEd.
        step = eqxi.error_if(
            state.step,
            (state.step == 1) & (state.flip ^ (error > 0)),
            msg="The root is not contained in [lower, upper]",
        )
        new_state = _BisectionState(
            lower=new_lower,
            upper=new_upper,
            flip=flip,
            step=step + 1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        return jnp.abs(state.lower - state.upper) < scale, RESULTS.successful

    def buffers(self, state: _BisectionState) -> tuple[()]:
        return ()


Bisection.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
"""
