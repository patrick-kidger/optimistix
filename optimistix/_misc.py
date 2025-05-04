import inspect
from collections.abc import Callable
from typing import Any, Literal, overload, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.extend as jex
import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar, ScalarLike
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


@overload
def tree_where(
    pred: Bool[ArrayLike, ""], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    ...


@overload
def tree_where(
    pred: PyTree[ArrayLike], true: PyTree[ArrayLike], false: PyTree[ArrayLike]
) -> PyTree[Array]:
    ...


def tree_where(pred, true, false):
    """Return a pytree with values from `true` where `pred` is true, and `false` where
    `pred` is false. If `pred` is a single boolean, then the same `pred` is used for all
    elements of the tree.
    """
    if jtu.tree_structure(pred) == jtu.tree_structure(true):
        if jtu.tree_structure(true) == jtu.tree_structure(false):  # false is PyTree
            return jtu.tree_map(lambda p, t, f: jnp.where(p, t, f), pred, true, false)
        else:
            return jtu.tree_map(lambda p, t: jnp.where(p, t, false), pred, true)
    else:  # pred is a boolean
        if jtu.tree_structure(true) == jtu.tree_structure(false):
            return jtu.tree_map(lambda t, f: jnp.where(pred, t, f), true, false)
        else:  # false is not a PyTree
            # TODO assert that false is a Scalar?
            return jtu.tree_map(lambda t: jnp.where(pred, t, false), true)


def tree_clip(
    tree: PyTree[ArrayLike], lower: PyTree[ArrayLike], upper: PyTree[ArrayLike]
) -> PyTree[Array]:
    """Clip tree to values lower, upper. Note that we do not check that the lower bound
    is actually less than the upper bound. If the upper bound is less than the lower
    bound, then the values will be clipped to the upper bound, since this is what
    `jnp.clip` does.
    """
    return jtu.tree_map(lambda x, l, u: jnp.clip(x, min=l, max=u), tree, lower, upper)


def tree_min(tree: PyTree[ArrayLike]) -> Scalar:
    values, _ = jfu.ravel_pytree(tree)
    return jnp.min(values)


def feasible_step_length(
    current: PyTree[Array],
    proposed_step: PyTree[Array],
    lower_bound: PyTree[Array],
    upper_bound: PyTree[Array],
    *,
    offset: ScalarLike = jnp.array(0.0),
    reduce: bool = True,
) -> Union[ScalarLike, PyTree[ScalarLike]]:
    """Returns the maximum feasible step length for any current value, its bounds, and a
    proposed step. The current value can be an instance of an optimisation variable `y`,
    a dual variable, or a slack variable.

    If the proposed step has a positive sign, then it is limited by the distance to the
    upper bound. If the proposed step has a negative sign, then it is limited by the
    distance to the lower bound. When we're at the boundary or outside of it, steps that
    improve upon this are allowed - e.g. when (upper - y) is negative, then we may take
    a step in the direction of -y. Similarly, if (upper - y) is zero, steps in the
    direction of -y are allowed.
    As such, this utility function does not check whether the current value is within
    the bounds and does not raise an error if it is not. It will, however, not make the
    problem worse either. This means that it is up to the  solvers how this case is
    handled or prevented.
    (In certain active set methods, when values are allowed to be at the bounds, small
    deviations outside of the bounds may be due to numerical errors.)

    The distance to the bounds is computed as (1 - offset) * (x - lower) for the lower
    bound, and (1 - offset) * (upper - x) for the upper bound. The offset can be used to
    ensure that the step taken remains in the strict interior of the feasible set. (This
    is required for interior point methods, where e.g. slack variables are required to
    be strictly positive.)
    Note that this function can return a feasible step length of 0.0, and this case
    should be handled where this function is used.

    **Arguments**:

    - `current`: The current value of an optimisation variable, e.g. `y`.
    - `proposed_step`: The proposed step - usually computed as the result of a linear
        solve. Must have the same PyTree structure as `current`.
    - `lower`: The lower bound. Must have the same PyTree structure as `current`.
    - `upper`: The upper bound. Must have the same PyTree structure as `current`.
    - `offset`: The offset from the boundary. If passed, then the distance to the bounds
        is multiplied by (1 - offset), to ensure that we stay in the strict interior.
        Keyword-only argument.
    - `reduce`: Whether to reduce the computed maximum step length for each leaf of the
        PyTree. (An array counts as one leaf.) This is useful in algorithms that allow
        for different step lengths in different variables, e.g. primal variables `y` and
        Lagrangian multipliers. Keyword-only argument, defaults to True - in which case
        a single scalar value is returned.
    """
    # TODO: runtime error if lower > upper for any element?
    # TODO: return a boolean array to indicate whether the feasible step length is > 0?

    def max_step(x, dx, lower, upper):
        distance_to_lower = (1 - offset) * (x - lower)
        distance_to_upper = (1 - offset) * (upper - x)

        # Scale by the distance to the bounds if we're moving towards them
        max_to_lower = jnp.asarray(jnp.where(dx < 0, -distance_to_lower / dx, jnp.inf))
        max_to_upper = jnp.asarray(jnp.where(dx > 0, distance_to_upper / dx, jnp.inf))

        # Negative distances when we're outside the bounds can result in step size < 0
        # this means that we would worsen our current position, which we don't want
        nonnegative_max_to_lower = jnp.where(max_to_lower > 0, max_to_lower, 0.0)
        nonnegative_max_to_upper = jnp.where(max_to_upper > 0, max_to_upper, 0.0)

        min_to_lower = jnp.min(jnp.asarray(nonnegative_max_to_lower))
        min_to_upper = jnp.min(jnp.asarray(nonnegative_max_to_upper))

        return jnp.min(jnp.array([min_to_lower, min_to_upper, 1.0]))

    max_steps = jtu.tree_map(max_step, current, proposed_step, lower_bound, upper_bound)

    if reduce:
        tree_min(max_steps)
    else:
        return max_steps


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


class OutAsArray(eqx.Module, strict=True):
    """Wrap a function that returns two arguments so that its mathematical outputs are
    all inexact arrays, and its auxiliary outputs are all arrays.
    For a minimisation function, the first return argument is the function value, and
    the second argument is any auxiliary output. For a constraint function, the first
    argument are the evaluated equality constraints, and the second argument are the
    evaluated inequality constraints.
    """

    fn: Callable

    def __call__(self, *args, **kwargs):
        first, second = self.fn(*args, **kwargs)
        first = jtu.tree_map(inexact_asarray, first)
        second = jtu.tree_map(asarray, second)
        return first, second


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


def checked_bounds(y: Y, bounds: tuple[Y, Y]) -> tuple[Y, Y]:
    msg = (
        "The PyTree structure of the bounds does not match the structure of `y0`."
        "Got structures {} and {}, but expected structure {} for both."
    )
    lower, upper = bounds
    lower_struct = jtu.tree_structure(lower)
    upper_struct = jtu.tree_structure(upper)
    y_struct = jtu.tree_structure(y)
    if not lower_struct == y_struct or not upper_struct == y_struct:
        raise ValueError(msg.format(lower_struct, upper_struct, y_struct))

    return bounds


def scalarlike_asarray(x: ScalarLike) -> Array:
    # https://github.com/patrick-kidger/optimistix/pull/109#issuecomment-2645950275
    return jnp.asarray(x)


# TODO: typing
def evaluate_constraint(constraint, y):
    assert constraint is not None

    constraint_residual = constraint(y)
    constraint_bound = constraint(tree_full_like(y, 0))

    equality_residual, inequality_residual = constraint_residual
    if equality_residual is not None:
        equality_jacobian, _ = jax.jacfwd(constraint)(y)
        equality_jac = lx.PyTreeLinearOperator(
            equality_jacobian, jax.eval_shape(lambda: equality_residual)
        )
    else:
        equality_jac = None

    if inequality_residual is not None:
        _, inequality_jacobian = jax.jacfwd(constraint)(y)
        inequality_jac = lx.PyTreeLinearOperator(
            inequality_jacobian, jax.eval_shape(lambda: inequality_residual)
        )
    else:
        inequality_jac = None
    return constraint_residual, constraint_bound, (equality_jac, inequality_jac)
