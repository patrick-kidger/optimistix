import functools as ft
import math
from typing import Callable

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PyTree, Shaped

from .custom_types import Scalar


def rms_norm(x: PyTree) -> Scalar:
    x, _ = jfu.ravel_pytree(x)
    if x.size == 0:
        return 0
    return _rms_norm(x)


@jax.custom_jvp
def _rms_norm(x):
    x_sq = jnp.real(x * jnp.conj(x))
    return jnp.sqrt(jnp.mean(x_sq))


@_rms_norm.defjvp
def _rms_norm_jvp(x, tx):
    (x,) = x
    (tx,) = tx
    out = _rms_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases
    pred = (out == 0) | jnp.isinf(out)
    numerator = jnp.where(pred, 0, x)
    denominator = jnp.where(pred, 1, out * x.size)
    t_out = jnp.dot(numerator / denominator, tx)
    return out, t_out


def max_norm(x: PyTree) -> Scalar:
    return max([jnp.max(jnp.abs(xi)) for xi in jtu.tree_leaves(x)])


def resolve_rcond(rcond, n, m, dtype):
    if rcond is None:
        return jnp.finfo(dtype).eps * max(n, m)
    else:
        return jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)


class NoneAux(eqx.Module):
    fn: Callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs), None


def jacobian(fn, in_size, out_size, has_aux=False):
    # Heuristic for which is better in each case
    # These could probably be tuned a lot more.
    if (in_size < 100) or (in_size <= 1.5 * out_size):
        return jax.jacfwd(fn, has_aux=has_aux)
    else:
        return jax.jacrev(fn, has_aux=has_aux)


def _to_struct(x):
    if eqx.is_array(x):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)
    else:
        return x


@ft.lru_cache(maxsize=128)
def _cached_eval_shape(leaves, treedef):
    fn, args, kwargs = jtu.tree_unflatten(treedef, leaves)
    return eqx.filter_eval_shape(fn, *args, **kwargs)


def cached_eval_shape(fn, *args, **kwargs):
    tree = jtu.tree_map(_to_struct, (fn, args, kwargs))
    leaves, treedef = jtu.tree_flatten(tree)
    leaves = tuple(leaves)
    return _cached_eval_shape(leaves, treedef)


def ordered_ravel(pytree: PyTree[Array], structure: PyTree[jax.ShapeDtypeStruct]):
    # `is` in case `tree_equal` returns a Tracer.
    if eqx.tree_equal(jax.eval_shape(lambda: pytree), structure) is not True:
        raise ValueError("pytree does not match structure")
    # not using `ravel_pytree` as that doesn't come with guarantees about order
    leaves = jtu.tree_leaves(pytree)
    dtype = jnp.result_type(*leaves)
    return jnp.concatenate([x.astype(dtype).reshape(-1) for x in leaves])


def ordered_unravel(
    vector: Shaped[Array, " size"], structure: PyTree[jax.ShapeDtypeStruct]
):
    leaves, treedef = jtu.tree_flatten(structure)
    sizes = np.cumsum([math.prod(x.shape) for x in leaves[:-1]])
    split = jnp.split(vector, sizes)
    assert len(split) == len(leaves)
    shaped = [x.reshape(y.shape).astype(y.dtype) for x, y in zip(split, leaves)]
    return jtu.tree_unflatten(treedef, shaped)


def inexact_asarray(x):
    x = jnp.asarray(x)
    if not jnp.issubdtype(x, jnp.inexact):
        x = x.astype(jnp.float32)
    return x
