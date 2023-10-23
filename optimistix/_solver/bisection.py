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
from typing import Any, ClassVar, Literal, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from .._custom_types import Aux, Fn
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


class _BisectionState(eqx.Module):
    lower: Scalar
    upper: Scalar
    flip: Bool[Array, ""]
    error: Float[Array, ""]


class Bisection(AbstractRootFinder[Scalar, Scalar, Aux, _BisectionState]):
    """The bisection method of root finding. This may only be used with functions
    `R->R`, i.e. functions with scalar input and scalar output.

    This requires the following `options`:

    - `lower`: The lower bound on the interval which contains the root.
    - `upper`: The upper bound on the interval which contains the root.

    Which are passed as, for example,
    `optimistix.root_find(..., options=dict(lower=0, upper=1))`

    This algorithm works by considering the interval `[lower, upper]`, checking the
    sign of the evaluated function at the midpoint of the interval, and then keeping
    whichever half contains the root. This is then repeated. The iteration stops once
    the interval is sufficiently small.
    """

    rtol: float
    atol: float
    flip: Union[bool, Literal["detect"]] = "detect"
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
        lower = jnp.asarray(options["lower"], f_struct.dtype)
        upper = jnp.asarray(options["upper"], f_struct.dtype)
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
        if isinstance(self.flip, bool):
            # Make it possible to avoid the extra two function compilations.
            flip = jnp.array(self.flip)
        elif self.flip == "detect":
            lower_val, _ = fn(lower, args)
            upper_val, _ = fn(upper, args)
            lower_neg = lower_val < 0
            upper_neg = upper_val < 0
            flip = lower_val > upper_val
            flip = eqx.error_if(
                flip,
                lower_neg == upper_neg,
                msg="The root is not contained in [lower, upper]",
            )
        else:
            raise ValueError("`flip` may only be True, False, or 'detect'.")
        return _BisectionState(
            lower=lower,
            upper=upper,
            flip=flip,
            error=jnp.array(jnp.inf, f_struct.dtype),
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
        del options
        error, aux = fn(y, args)
        negative = state.flip ^ (error < 0)
        new_lower = jnp.where(negative, y, state.lower)
        new_upper = jnp.where(negative, state.upper, y)
        new_y = new_lower + 0.5 * (new_upper - new_lower)
        new_state = _BisectionState(
            lower=new_lower, upper=new_upper, flip=state.flip, error=error
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
        y_small = jnp.abs(state.lower - state.upper) < scale
        f_small = jnp.abs(state.error) < self.atol
        return y_small & f_small, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Scalar, Aux, dict[str, Any]]:
        return y, aux, {}


Bisection.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `flip`: Can be set to any of:
    - `False`: specify that `fn(lower, args) < 0 < fn(upper, args)`.
    - `True`: specify that `fn(lower, args) > 0 > fn(upper, args)`.
    - `"detect"`: automatically check `fn(lower, args)` and `fn(upper, args)`. Note that
        this option may increase both runtime and compilation time.
"""
