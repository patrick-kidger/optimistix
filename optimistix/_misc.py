import inspect
from collections.abc import Callable
from typing import Any, Literal, overload, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.extend as jex
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar
from lineax.internal import (
    default_floating_dtype as _default_floating_dtype,
    max_norm as _max_norm,
    rms_norm as _rms_norm,
    sum_squares as _sum_squares,
    tree_dot as _tree_dot,
    two_norm as _two_norm,
)

from ._custom_types import Y


# Make the wrapped function a genuine member of this module.
def _wrap(fn):
    # Not using `functools.wraps` as our docgen will chase that.
    def wrapped_fn(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapped_fn.__signature__ = inspect.signature(fn)
    wrapped_fn.__name__ = wrapped_fn.__qualname__ = fn.__name__
    wrapped_fn.__module__ = __name__
    wrapped_fn.__doc__ = fn.__doc__
    return wrapped_fn


default_floating_dtype = _wrap(_default_floating_dtype)
max_norm = _wrap(_max_norm)
rms_norm = _wrap(_rms_norm)
sum_squares = _wrap(_sum_squares)
tree_dot = _wrap(_tree_dot)
two_norm = _wrap(_two_norm)


@overload
def tree_full_like(
    struct: PyTree[Array | jax.ShapeDtypeStruct],
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


def lin_to_grad(lin_fn, y_eval, autodiff_mode=None):
    # Only the shape and dtype of y_eval is evaluated, not the value itself. (lin_fn
    # was linearized at y_eval, and the values were stored.)
    # We convert to grad after linearising for efficiency:
    # https://github.com/patrick-kidger/optimistix/issues/89#issuecomment-2447669714
    if autodiff_mode == "bwd":
        (grad,) = jax.linear_transpose(lin_fn, y_eval)(1.0)  # (1.0 is a scaling factor)
        return grad
    if autodiff_mode == "fwd":
        return jax.jacfwd(lin_fn)(y_eval)
    else:
        raise ValueError(
            "Only `autodiff_mode='fwd'` or `autodiff_mode='bwd'` are valid."
        )


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
