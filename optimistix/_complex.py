from typing import Any, Generic

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from ._custom_types import Args, Aux, Fn, Out, Y
from ._solution import Solution


class _RealPair(eqx.Module):
    real: Array
    imag: Array


class _FromRealFn(eqx.Module, Generic[Y, Out, Aux]):
    fn: Fn[Y, Out, Aux]
    convert_output: bool = eqx.field(static=True)

    def __call__(self, y: PyTree, args: Args) -> tuple[Out, Aux]:
        out, aux = self.fn(_real_to_complex(y), args)
        if self.convert_output:
            out = _complex_to_real(out)
        return out, aux


def _is_complex_leaf(x: Any) -> bool:
    return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.complexfloating)


def _has_complex(x: PyTree) -> bool:
    return any(_is_complex_leaf(leaf) for leaf in jtu.tree_leaves(x))


def _complex_to_real(x: PyTree) -> PyTree:
    def _to_real_pair(leaf):
        if _is_complex_leaf(leaf):
            return _RealPair(leaf.real, leaf.imag)
        return leaf

    return jtu.tree_map(_to_real_pair, x)


def _real_to_complex(x: PyTree) -> PyTree:
    def _to_complex(leaf):
        if isinstance(leaf, _RealPair):
            return lax.complex(leaf.real, leaf.imag)
        return leaf

    return jtu.tree_map(
        _to_complex, x, is_leaf=lambda leaf: isinstance(leaf, _RealPair)
    )


def _restore_solution(solution: Solution) -> Solution:
    value = _real_to_complex(solution.value)
    return eqx.tree_at(lambda sol: sol.value, solution, value)
