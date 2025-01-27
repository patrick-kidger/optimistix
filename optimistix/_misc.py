from collections.abc import Callable
from typing import Any, Literal, overload, TypeVar, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.extend as jex
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar
from lineax.internal import (
    default_floating_dtype as default_floating_dtype,
    max_norm as max_norm,
    rms_norm as rms_norm,
    sum_squares as sum_squares,
    tree_dot as tree_dot,
    two_norm as two_norm,
)

from ._custom_types import Y


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


def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    """Return the `true` or `false` pytree depending on `pred`."""
    keep = lambda a, b: jnp.where(pred, a, b)
    return jtu.tree_map(keep, true, false)


def resolve_rcond(rcond, n, m, dtype):
    if rcond is None:
        return jnp.finfo(dtype).eps * max(n, m)
    else:
        return jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)


class NoneAux(eqx.Module, strict=True):
    """Wrap a function `fn` so it returns a dummy aux value `None`

    NoneAux is used to give a consistent API between functions which have an aux
    and functions which do not, allowing us to avoid unnecessary aux handling.
    """

    fn: Callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs), None


class OutAsArray(eqx.Module, strict=True):
    """Wrap a minimisation/root-find/etc. function so that its mathematical outputs are
    all inexact arrays, and its auxiliary outputs are all arrays.
    """

    fn: Callable

    def __call__(self, *args, **kwargs):
        out, aux = self.fn(*args, **kwargs)
        out = jtu.tree_map(inexact_asarray, out)
        aux = jtu.tree_map(asarray, aux)
        return out, aux


def _jacfwd(lin_fn, pytree):
    """Custom version of jax.jacfwd that directly uses a linearized function.
    Takes inspiration from jax.jacfwd, but simplifies some steps: we only ever treat
    PyTrees of arrays, where all elements have the same dtype.
    """
    leaves, treedef = jtu.tree_flatten(pytree)
    static_sizes = [int(jnp.size(leaf)) for leaf in leaves]
    indices = np.cumsum(static_sizes)[:-1]  # Define boundaries between elements
    elements = sum(static_sizes)
    dtype = jax.dtypes.result_type(*leaves)

    def values_to_tree(values):
        parts = jnp.split(values, indices)
        reshaped = jtu.tree_map(lambda a, b: jnp.reshape(a, b.shape), parts, leaves)
        return jtu.tree_unflatten(treedef, reshaped)

    def unit_tree(index):
        values = jnp.zeros(elements, dtype=dtype).at[index].set(1.0)
        return values_to_tree(values)

    unit_pytrees = [unit_tree(i) for i in range(elements)]
    derivatives = jnp.stack([lin_fn(t) for t in unit_pytrees])
    return values_to_tree(derivatives)


def lin_to_grad(lin_fn, y_eval, mode=None):
    # Only the shape and dtype of y_eval is evaluated, not the value itself. (lin_fn
    # was linearized at y_eval, and the values were stored.)
    # We convert to grad after linearising for efficiency:
    # https://github.com/patrick-kidger/optimistix/issues/89#issuecomment-2447669714
    if mode == "bwd":
        (grad,) = jax.linear_transpose(lin_fn, y_eval)(1.0)  # (1.0 is a scaling factor)
        return grad
    if mode == "fwd":
        return _jacfwd(lin_fn, y_eval)
    else:
        raise ValueError("Only `mode='fwd'` or `mode='bwd'` are valid.")


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
    def __init__(self, jaxpr: jex.core.Jaxpr):
        self.jaxpr = jaxpr

    def __hash__(self):
        return 0

    def __eq__(self, other):
        # Could also implement actual checks here.
        return type(self) is type(other)


def _wrap_jaxpr_leaf(leaf):
    # But not jex.core.ClosedJaxpr, which contains constants that should be handled in
    # pytree style.
    if isinstance(leaf, jex.core.Jaxpr):
        return _JaxprEqual(leaf)
    else:
        return leaf


def wrap_jaxpr(tree):
    return jtu.tree_map(_wrap_jaxpr_leaf, tree)


def _unwrap_jaxpr_leaf(leaf):
    if isinstance(leaf, _JaxprEqual):
        return leaf.jaxpr
    else:
        return leaf


def unwrap_jaxpr(tree):
    return jtu.tree_map(_unwrap_jaxpr_leaf, tree)


def filter_cond(pred, true_fun, false_fun, *operands):
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, static_out.value)


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
