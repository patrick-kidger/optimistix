from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._fixed_point import AbstractFixedPointSolver
from .._misc import max_norm
from .._solution import RESULTS


class _FixedPointState(eqx.Module):
    relative_error: Scalar


class FixedPointIteration(AbstractFixedPointSolver[Y, Aux, _FixedPointState]):
    """Repeatedly calls a function in search of a fixed point.

    This is one of the simplest ways to find a fixed point `y` of `f`: simply
    repeatedly call `y_{n+1}=f(y_n)` until `y_n` stops changing.

    Optionally, one can use damping between iterations, then the update becomes:
    `y_{n+1}=(1-damp)*f(y_n)+damp*y_n`
    Where damp is the damping factor.

    Note that this is often not a very effective method, and root-finding algorithms are
    frequently preferred in practice.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    damp: float = 0.0

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
        fn_y_n, aux = fn(y, args)
        new_y = jtu.tree_map(lambda i_np1, i_n : (1-self.damp)*i_np1+self.damp*i_n, fn_y_n ,y)
        error = (y**ω - new_y**ω).ω
        with jax.numpy_dtype_promotion("standard"):
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

    def postprocess(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


FixedPointIteration.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `damp`: The damping factor used in iteration update.
"""
