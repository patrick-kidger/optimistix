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
from typing import Any, Literal, overload, TYPE_CHECKING, TypeVar, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, Inexact, PyTree, Scalar

from ._custom_types import Y


if TYPE_CHECKING:
    pass
else:
    pass


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


@overload
def tree_full_like(
    struct: PyTree[Union[Array, jax.ShapeDtypeStruct]],
    fill_value: ArrayLike,
    allow_static: Literal[False] = False,
):
    ...


@overload
def tree_full_like(
    struct: PyTree, fill_value: ArrayLike, allow_static: Literal[True] = True
):

    ...


def tree_full_like(struct: PyTree, fill_value: ArrayLike, allow_static: bool = False):
    """Return a pytree with the same type and shape as the input with values
    `fill_value`.

    If `allow_static=True`, then any non-{array, struct}s are ignored and left alone.
    If `allow_static=False` then any non-{array, struct}s will result in an error.
    """
    fn = lambda x: jnp.full(x.shape, fill_value, x.dtype)
    if isinstance(fill_value, (int, float)):
        if fill_value == 0:
            fn = lambda x: jnp.zeros(x.shape, x.dtype)
        elif fill_value == 1:
            fn = lambda x: jnp.ones(x.shape, x.dtype)
    if allow_static:
        _fn = fn
        fn = (
            lambda x: _fn(x)
            if eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)
            else x
        )
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
                precision=jax.lax.Precision.HIGHEST,  # pyright: ignore
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
    leaves = jtu.tree_leaves(x)
    size = sum([jnp.size(xi) for xi in leaves])
    if size == 1:
        # Avoid needless squaring-and-then-rooting.
        for leaf in leaves:
            if jnp.size(leaf) == 1:
                return jnp.abs(jnp.reshape(leaf, ()))
        else:
            assert False
    else:
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
    size = sum([jnp.size(xi) for xi in leaves])
    return two_norm(x) / math.sqrt(size)


def max_norm(x: PyTree[ArrayLike]) -> Scalar:
    """Compute the L-infinity norm of a PyTree of arrays.

    This is the largest absolute elementwise value. Considering the input `x` as a flat
    vector `(x_1, ..., x_n)`, then this computes `max_i |x_i|`.
    """
    leaf_maxes = [jnp.max(jnp.abs(xi)) for xi in jtu.tree_leaves(x) if jnp.size(xi) > 0]
    if len(leaf_maxes) == 0:
        return jnp.array(0, default_floating_dtype())
    else:
        out = ft.reduce(jnp.maximum, leaf_maxes)
        return _zero_grad_at_zero(out)


@jax.custom_jvp
def _zero_grad_at_zero(x):
    return x


@_zero_grad_at_zero.defjvp
def _zero_grad_at_zero_jvp(primals, tangents):
    (out,) = primals
    (t_out,) = tangents
    return out, jnp.where(out == 0, 0, t_out)


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


class OutAsArray(eqx.Module):
    """Wrap a minimisation/root-find/etc. function so that its mathematical outputs are
    all inexact arrays, and its auxiliary outputs are all arrays.
    """

    fn: Callable

    def __call__(self, *args, **kwargs):
        out, aux = self.fn(*args, **kwargs)
        out = jtu.tree_map(inexact_asarray, out)
        aux = jtu.tree_map(asarray, aux)
        return out, aux


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


def lin_to_grad(lin_fn, *primals):
    return jax.linear_transpose(lin_fn, *primals)(1.0)


def _asarray(dtype, x):
    return jnp.asarray(x, dtype=dtype)


# Work around JAX issue #15676
_asarray = jax.custom_jvp(_asarray, nondiff_argnums=(0,))


@_asarray.defjvp
def _asarray_jvp(dtype, x, tx):
    (x,) = x
    (tx,) = tx
    return _asarray(dtype, x), _asarray(dtype, tx)


def asarray(x):
    dtype = jnp.result_type(x)
    return _asarray(dtype, x)


def inexact_asarray(x):
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        dtype = default_floating_dtype()
    return _asarray(dtype, x)


_F = TypeVar("_F")


def cauchy_termination(
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], Scalar],
    y: Y,
    y_diff: Y,
    f: _F,
    f_diff: _F,
) -> Bool[Array, ""]:
    """Terminate if there is a small difference in both `y` space and `f` space, as
    determined by `rtol` and `atol`.

    Specifically, this checks that `y_diff < atol + rtol * y` and
    `f_diff < atol + rtol * f_prev`, terminating if both of these are true.
    """
    y_scale = (atol + rtol * ω(y).call(jnp.abs)).ω
    f_scale = (atol + rtol * ω(f).call(jnp.abs)).ω
    y_converged = norm((ω(y_diff).call(jnp.abs) / y_scale**ω).ω) < 1
    f_converged = norm((ω(f_diff).call(jnp.abs) / f_scale**ω).ω) < 1
    return y_converged & f_converged


class _JaxprEqual:
    def __init__(self, jaxpr: jax.core.Jaxpr):
        self.jaxpr = jaxpr

    def __hash__(self):
        return 0

    def __eq__(self, other):
        # Could also implement actual checks here.
        return type(self) is type(other)


def _wrap_jaxpr_impl(leaf):
    # But not jax.core.ClosedJaxpr, which contains constants that should be handled in
    # pytree style.
    if isinstance(leaf, jax.core.Jaxpr):
        return _JaxprEqual(leaf)
    else:
        return leaf


def _wrap_jaxpr(tree):
    return jtu.tree_map(_wrap_jaxpr_impl, tree)


def _unwrap_jaxpr_impl(leaf):
    if isinstance(leaf, _JaxprEqual):
        return leaf.jaxpr
    else:
        return leaf


def _unwrap_jaxpr(tree):
    return jtu.tree_map(_unwrap_jaxpr_impl, tree)


def filter_cond(pred, true_fun, false_fun, *operands):
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        _static_out = _wrap_jaxpr(_static_out)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        _static_out = _wrap_jaxpr(_static_out)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, _unwrap_jaxpr(static_out.value))


def verbose_print(*args: tuple[bool, str, Any]) -> None:
    string_pieces = []
    arg_pieces = []
    for display, name, value in args:
        if display:
            string_pieces.append(name + ": {}")
            arg_pieces.append(value)
    if len(string_pieces) > 0:
        string = ", ".join(string_pieces)
        jax.debug.print(string, *arg_pieces)
