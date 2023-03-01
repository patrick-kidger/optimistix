import functools as ft
import operator
import random

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

import optimistix as optx


def getkey():
    return jr.PRNGKey(random.randint(0, 2**31 - 1))


def _shaped_allclose(x, y, **kwargs):
    if type(x) is not type(y):
        return False
    if isinstance(x, jnp.ndarray):
        if jnp.issubdtype(x.dtype, jnp.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and jnp.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and jnp.all(x == y)
    elif isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and np.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and np.all(x == y)
    else:
        return x == y


def shaped_allclose(x, y, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jtu.tree_structure(x) == jtu.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, **kwargs)
    return same_structure and jtu.tree_reduce(
        operator.and_, jtu.tree_map(allclose, x, y), True
    )


def has_tag(tags, tag):
    return tag is tags or (isinstance(tags, tuple) and tag in tags)


make_operators = []


def _operators_append(x):
    make_operators.append(x)
    return x


@_operators_append
def make_matrix_operator(matrix, tags):
    return optx.MatrixLinearOperator(matrix, tags)


@_operators_append
def make_trivial_pytree_operator(matrix, tags):
    struct = jax.ShapeDtypeStruct((3,), matrix.dtype)
    return optx.PyTreeLinearOperator(matrix, struct, tags)


@_operators_append
def make_function_operator(matrix, tags):
    fn = lambda x: matrix @ x
    in_struct = jax.ShapeDtypeStruct((3,), matrix.dtype)
    return optx.FunctionLinearOperator(fn, in_struct, tags)


@_operators_append
def make_jac_operator(matrix, tags):
    x = jr.normal(getkey(), (3,))
    b = jr.normal(getkey(), (3, 3))
    fn_tmp = lambda x, _: x + b @ x**2
    jac = jax.jacfwd(fn_tmp)(x, None)
    diff = matrix - jac
    fn = lambda x, _: x + b @ x**2 + diff @ x
    return optx.JacobianLinearOperator(fn, x, None, tags)


@_operators_append
def make_diagonal_operator(matrix, tags):
    assert has_tag(tags, optx.diagonal_tag)
    diag = jnp.diag(matrix)
    return optx.DiagonalLinearOperator(diag)


@_operators_append
def make_add_operator(matrix, tags):
    matrix1 = 0.7 * matrix
    matrix2 = 0.3 * matrix
    operator = make_matrix_operator(matrix1, ()) + make_function_operator(matrix2, ())
    return optx.TaggedLinearOperator(operator, tags)


@_operators_append
def make_mul_operator(matrix, tags):
    operator = make_jac_operator(0.7 * matrix, ()) / 0.7
    return optx.TaggedLinearOperator(operator, tags)


@_operators_append
def make_composed_operator(matrix, tags):
    size, _ = matrix.shape
    diag = jr.normal(getkey(), (size,))
    diag = jnp.where(jnp.abs(diag) < 0.05, 0.8, diag)
    operator1 = make_trivial_pytree_operator(matrix / diag, ())
    operator2 = optx.DiagonalLinearOperator(diag)
    return optx.TaggedLinearOperator(operator1 @ operator2, tags)
