from collections.abc import Callable
from typing import Any, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Constraint, EqualityOut, Fn, InequalityOut, Y
from .._fixed_point import AbstractFixedPointSolver
from .._misc import max_norm
from .._search import Iterate
from .._solution import RESULTS


class _FixedPointState(eqx.Module):
    relative_error: Scalar


class FixedPointIteration(
    AbstractFixedPointSolver[Y, Iterate.Primal, Aux, _FixedPointState],
):
    """Repeatedly calls a function in search of a fixed point.

    This is one of the simplest ways to find a fixed point `y` of `f`: simply
    repeatedly call `y_{n+1}=f(y_n)` until `y_n` stops changing.

    Note that this is often not a very effective method, and root-finding algorithms are
    frequently preferred in practice.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm

    def init(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> tuple[Iterate.Primal, _FixedPointState]:
        del fn, args, options, constraint, bounds, f_struct, aux_struct
        return Iterate.Primal(y), _FixedPointState(jnp.array(jnp.inf))

    def step(
        self,
        fn: Fn[Y, Y, Aux],
        iterate: Iterate.Primal,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Iterate.Primal, _FixedPointState, Aux]:
        y = iterate.y
        new_y, aux = fn(y, args)
        error = (y**ω - new_y**ω).ω
        with jax.numpy_dtype_promotion("standard"):
            scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
            new_state = _FixedPointState(self.norm((error**ω / scale**ω).ω))
        return Iterate.Primal(new_y), new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Y, Aux],
        iterate: Iterate.Primal,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.relative_error < 1, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Y, Aux],
        iterate: Iterate.Primal,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        state: _FixedPointState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return iterate.y, aux, {}


FixedPointIteration.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
"""
