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
from typing import Any, cast
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Scalar

from .._base_solver import AbstractHasTol
from .._custom_types import Aux, Fn, Y
from .._minimise import AbstractMinimiser
from .._misc import cauchy_termination, max_norm
from .._solution import RESULTS


_OptaxClass: TypeAlias = Any


class _OptaxState(eqx.Module):
    f: Scalar
    opt_state: Any
    terminate: Bool[Array, ""]


class OptaxMinimiser(AbstractMinimiser[Y, Aux, _OptaxState], AbstractHasTol):
    """A wrapper to use Optax first-order gradient-based optimisers with
    [`optimistix.minimise`][].
    """

    optax_cls: _OptaxClass
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]

    def __init__(
        self,
        optax_cls,
        *args,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        **kwargs
    ):
        """**Arguments:**

        - `optax_cls`: The **class** of the Optax method to use. Do not pass an
            **instance** of the Optax class.
        - `args`: The arguments used to instantiate `optax_cls`.
        - `kwargs`: The keyword arguments used to instantiate `optax_cls`.
        - `rtol`: Relative tolerance for terminating the solve. Keyword only argument.
        - `atol`: Absolute tolerance for terminating the solve. Keyword only argument.
        - `norm`: The norm used to determine the difference between two iterates in the
            convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
            includes three built-in norms: [`optimistix.max_norm`][],
            [`optimistix.rms_norm`][], and [`optimistix.two_norm`][]. Keyword only
            argument.
        """
        self.optax_cls = optax_cls
        self.args = args
        self.kwargs = kwargs
        self.rtol = rtol
        self.atol = atol
        self.norm = norm

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _OptaxState:
        del fn, args, options, aux_struct
        optim = self.optax_cls(*self.args, **self.kwargs)
        opt_state = optim.init(y)
        maxval = jnp.array(jnp.finfo(f_struct.dtype).max, f_struct.dtype)
        return _OptaxState(f=maxval, opt_state=opt_state, terminate=jnp.array(False))

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _OptaxState,
        tags: frozenset[object],
    ) -> tuple[Y, _OptaxState, Aux]:
        del options
        (f, aux), grads = eqx.filter_value_and_grad(fn, has_aux=True)(y, args)
        f = cast(Array, f)
        optim = self.optax_cls(*self.args, **self.kwargs)
        updates, new_opt_state = optim.update(grads, state.opt_state)
        new_y = eqx.apply_updates(y, updates)
        terminate = cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            updates,
            f,
            f - state.f,
        )
        new_state = _OptaxState(f=f, opt_state=new_opt_state, terminate=terminate)
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _OptaxState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        return state.terminate, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _OptaxState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}
