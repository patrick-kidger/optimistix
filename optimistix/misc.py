import functools as ft
from typing import Callable

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool, PyTree

from .custom_types import Scalar


def two_norm(x: PyTree) -> Scalar:
    x, _ = jfu.ravel_pytree(x)
    if x.size == 0:
        return 0
    return _two_norm(x)


@jax.custom_jvp
def _two_norm(x):
    x_sq = jnp.real(x * jnp.conj(x))
    return jnp.sqrt(jnp.sum(x_sq))


@_two_norm.defjvp
def _two_norm_jvp(x, tx):
    (x,) = x
    (tx,) = tx
    out = _two_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases
    pred = (out == 0) | jnp.isinf(out)
    numerator = jnp.where(pred, 0, jnp.dot(x, tx))
    denominator = jnp.where(pred, 1, out)
    t_out = numerator / denominator
    return out, t_out


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


def tree_inner_prod(tree1, tree2):
    tree1_ravel, _ = jax.flatten_util.ravel_pytree(tree1)
    tree2_ravel, _ = jax.flatten_util.ravel_pytree(tree2)
    return tree1_ravel.T @ tree2_ravel


def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)


def max_norm(x: PyTree) -> Scalar:
    leaf_maxes = [jnp.max(jnp.abs(xi)) for xi in jtu.tree_leaves(x)]
    return jtu.tree_reduce(jnp.maximum, leaf_maxes)


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


def default_floating_dtype():
    if jax.config.jax_enable_x64:
        return jnp.float64
    else:
        return jnp.float32


def inexact_asarray(x):
    x = jnp.asarray(x)
    if not jnp.issubdtype(x, jnp.inexact):
        x = x.astype(default_floating_dtype())
    return x
