from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._fixed_point import AbstractFixedPointSolver
from .._misc import max_norm
from .._solution import RESULTS


class _FixedPointState(eqx.Module):
    relative_error: Scalar


class FixedPointIteration(AbstractFixedPointSolver[_FixedPointState, Y, Aux]):
    rtol: float
    atol: float
    norm: Callable = max_norm

    def init(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _FixedPointState:
        del fn, y, args, options, f_struct, aux_struct
        return _FixedPointState(jnp.array(jnp.inf))

    def step(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Y, _FixedPointState, Aux]:
        new_y, aux = fn(y, args)
        error = (y**ω - new_y**ω).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        new_state = _FixedPointState(self.norm((error**ω / scale**ω).ω))
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.relative_error < 1, RESULTS.successful

    def buffers(self, state: _FixedPointState) -> tuple[()]:
        return ()
