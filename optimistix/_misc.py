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

import functools as ft
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, Inexact, PyTree, Scalar


if TYPE_CHECKING:
    pass
else:
    pass


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def tree_full_like(
    struct: PyTree[Union[Array, jax.ShapeDtypeStruct]], fill_value: ArrayLike
):
    """Return a pytree with the same type and shape as the input with values
    `fill_value`.
    """
    fn = lambda x: jnp.full(x.shape, fill_value, x.dtype)
    if isinstance(fill_value, (int, float)):
        if fill_value == 0:
            fn = lambda x: jnp.zeros(x.shape, x.dtype)
        elif fill_value == 1:
            fn = lambda x: jnp.ones(x.shape, x.dtype)
    return jtu.tree_map(fn, struct)


def tree_dot(tree1: PyTree[ArrayLike], tree2: PyTree[ArrayLike]) -> Inexact[Array, ""]:
    """Compute the dot product of two pytrees of arrays with the same pytree
    structure."""
    leaves1, treedef1 = jtu.tree_flatten(tree1)
    leaves2, treedef2 = jtu.tree_flatten(tree2)
    if treedef1 != treedef2:
        raise ValueError("trees must have the same structure")
    assert len(leaves1) == len(leaves2)
    dots = []
    for leaf1, leaf2 in zip(leaves1, leaves2):
        dots.append(
            jnp.dot(
                jnp.reshape(leaf1, -1),
                jnp.conj(leaf2).reshape(-1),
                precision=jax.lax.Precision.HIGHEST,
            )
        )
    if len(dots) == 0:
        return jnp.array(0, default_floating_dtype())
    else:
        return ft.reduce(jnp.add, dots)


def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    """Return the `true` or `false` pytree depending on `pred`."""
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)


def sum_squares(x: PyTree[ArrayLike]) -> Scalar:
    """Computes the square of the L2 norm of a PyTree of arrays.

    Considering the input `x` as a flat vector `(x_1, ..., x_n)`, then this computes
    `Σ_i x_i^2`
    """
    return tree_dot(x, x).real


@jax.custom_jvp
def two_norm(x: PyTree[ArrayLike]) -> Scalar:
    """Computes the L2 norm of a PyTree of arrays.

    Considering the input `x` as a flat vector `(x_1, ..., x_n)`, then this computes
    `sqrt(Σ_i x_i^2)`
    """
    return jnp.sqrt(sum_squares(x))


@two_norm.defjvp
def _two_norm_jvp(x, tx):
    (x,) = x
    (tx,) = tx
    out = two_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases.
    pred = (out == 0) | jnp.isinf(out)
    denominator = jnp.where(pred, 1, out)
    # We could also switch the dot and the division.
    # This approach is a bit more expensive (more divisions), but should be more
    # numerically stable (`x` and `denominator` should be of the same scale; `tx` is of
    # unknown scale).
    t_out = tree_dot((x**ω / denominator).ω, tx).real
    t_out = jnp.where(pred, 0, t_out)
    return out, t_out


def rms_norm(x: PyTree[ArrayLike]) -> Scalar:
    """Compute the RMS (root-mean-squared) norm of a PyTree of arrays.

    This is the same as the L2 norm, averaged by the size of the input `x`. Considering
    the input `x` as a flat vector `(x_1, ..., x_n)`, then this computes
    `sqrt((Σ_i x_i^2)/n)`
    """
    leaves = jtu.tree_leaves(x)
    size = sum([jnp.size(x) for x in leaves])
    return two_norm(x) / math.sqrt(size)


def max_norm(x: PyTree[ArrayLike]) -> Scalar:
    """Compute the L-infinity norm of a PyTree of arrays.

    This is the largest absolute elementwise value. Considering the input `x` as a flat
    vector `(x_1, ..., x_n)`, then this computes `max_i |x_i|`.
    """
    leaf_maxes = [jnp.max(jnp.abs(xi)) for xi in jtu.tree_leaves(x)]
    if len(leaf_maxes) == 0:
        return jnp.array(0, default_floating_dtype())
    else:
        return ft.reduce(jnp.maximum, leaf_maxes)


def resolve_rcond(rcond, n, m, dtype):
    if rcond is None:
        return jnp.finfo(dtype).eps * max(n, m)
    else:
        return jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)


class NoneAux(eqx.Module):
    """Wrap a function `fn` so it returns a dummy aux value `None`

    NoneAux is used to give a consistent API between functions which have an aux
    and functions which do not, allowing us to avoid unnecessary aux handling.
    """

    fn: Callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs), None


def jacobian(fn, in_size, out_size, has_aux=False):
    """Compute the Jacobian of a function using forward or backward mode AD.

    `jacobian` chooses between forward and backwards autodiff depending on the input
    and output dimension of `fn`, as specified in `in_size` and `out_size`.
    """

    # Heuristic for which is better in each case
    # These could probably be tuned a lot more.
    if (in_size < 100) or (in_size <= 1.5 * out_size):
        return jax.jacfwd(fn, has_aux=has_aux)
    else:
        return jax.jacrev(fn, has_aux=has_aux)


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


def is_linear(fn, *args, output):
    try:
        eqx.filter_eval_shape(
            lambda x, y: jax.linear_transpose(fn, *x)(y), args, output
        )
    except Exception:
        return False
    else:
        return True
