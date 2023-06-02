import functools as ft
from typing import Callable, cast, Optional

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree, Scalar


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
    x = cast(Array, x)
    tx = cast(Array, x)
    out = _rms_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases
    pred = (out == 0) | jnp.isinf(out)
    numerator = jnp.where(pred, 0, x)
    denominator = jnp.where(pred, 1, out * x.size)
    t_out = jnp.dot(numerator / denominator, tx)
    return out, t_out


def tree_full(struct: Optional[PyTree[jax.ShapeDtypeStruct]], fill_value: ArrayLike):
    if struct is None:
        filled = None
    else:
        filled = jtu.tree_map(lambda x: jnp.full(x.shape, fill_value, x.dtype), struct)
    return filled


def tree_full_like(tree: PyTree[Array], fill_value: ArrayLike):
    return jtu.tree_map(lambda x: jnp.full_like(x, fill_value), tree)


def tree_zeros(struct: Optional[PyTree[jax.ShapeDtypeStruct]]):
    if struct is None:
        zeros = None
    else:
        zeros = jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), struct)
    return zeros


def tree_zeros_like(tree: PyTree[Array]) -> PyTree[Array]:
    return jtu.tree_map(jnp.zeros_like, tree)


def tree_inner_prod(tree1: PyTree[Array], tree2: PyTree[Array]) -> Float[Array, ""]:
    prod = (tree1**ω * tree2**ω).call(jnp.sum).ω
    return jtu.tree_reduce(lambda x, y: x + y, prod)


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


def _asarray(dtype, x):
    return jnp.asarray(x, dtype=dtype)


# Work around JAX issue #15676
_asarray = jax.custom_jvp(_asarray, nondiff_argnums=(0,))


@_asarray.defjvp
def _asarray_jvp(dtype, x, tx):
    (x,) = x
    (tx,) = tx
    return _asarray(dtype, x), _asarray(dtype, tx)


def inexact_asarray(x):
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        dtype = default_floating_dtype()
    return _asarray(dtype, x)
