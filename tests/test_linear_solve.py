import math

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import optimistix as optx

from .helpers import getkey, shaped_allclose


_make_operators = []


def _make_operators_append(x):
    _make_operators.append(x)
    return x


@_make_operators_append
def _make_matrix_operator(matrix, tags):
    return optx.MatrixLinearOperator(matrix, tags)


@_make_operators_append
def _make_trivial_pytree_operator(matrix, tags):
    struct = jax.ShapeDtypeStruct((3,), matrix.dtype)
    return optx.PyTreeLinearOperator(matrix, struct, tags)


@_make_operators_append
def _make_function_operator(matrix, tags):
    fn = lambda x: matrix @ x
    in_struct = jax.ShapeDtypeStruct((3,), matrix.dtype)
    return optx.FunctionLinearOperator(fn, in_struct, tags)


@_make_operators_append
def _make_jac_operator(matrix, tags):
    x = jr.normal(getkey(), (3,))
    b = jr.normal(getkey(), (3, 3))
    fn_tmp = lambda x, _: x + b @ x**2
    jac = jax.jacfwd(fn_tmp)(x, None)
    diff = matrix - jac
    fn = lambda x, _: x + b @ x**2 + diff @ x
    return optx.JacobianLinearOperator(fn, x, None, tags)


@_make_operators_append
def _make_diagonal_operator(matrix, tags):
    assert _has(tags, optx.diagonal_tag)
    diag = jnp.diag(matrix)
    return optx.DiagonalLinearOperator(diag)


@_make_operators_append
def _make_add_operator(matrix, tags):
    matrix1 = 0.7 * matrix
    matrix2 = 0.3 * matrix
    operator = _make_matrix_operator(matrix1, ()) + _make_function_operator(matrix2, ())
    return optx.TaggedLinearOperator(operator, tags)


@_make_operators_append
def _make_mul_operator(matrix, tags):
    operator = _make_jac_operator(0.7 * matrix, ()) / 0.7
    return optx.TaggedLinearOperator(operator, tags)


@_make_operators_append
def _make_composed_operator(matrix, tags):
    size, _ = matrix.shape
    diag = jr.normal(getkey(), (size,))
    diag = jnp.where(jnp.abs(diag) < 0.05, 0.8, diag)
    operator1 = _make_trivial_pytree_operator(matrix / diag, ())
    operator2 = optx.DiagonalLinearOperator(diag)
    return optx.TaggedLinearOperator(operator1 @ operator2, tags)


if jax.config.jax_enable_x64:
    tol = 1e-12
else:
    tol = 1e-6
_solvers = [
    (optx.AutoLinearSolver(), ()),
    (optx.Triangular(), optx.lower_triangular_tag),
    (optx.Triangular(), optx.upper_triangular_tag),
    (optx.Triangular(), (optx.lower_triangular_tag, optx.unit_diagonal_tag)),
    (optx.Triangular(), (optx.upper_triangular_tag, optx.unit_diagonal_tag)),
    (optx.Diagonal(), optx.diagonal_tag),
    (optx.Diagonal(), (optx.diagonal_tag, optx.unit_diagonal_tag)),
    (optx.LU(), ()),
    (optx.QR(), ()),
    (optx.SVD(), ()),
    (optx.CG(normal=True, rtol=tol, atol=tol), ()),
    (optx.CG(normal=False, rtol=tol, atol=tol), optx.positive_semidefinite_tag),
    (optx.Cholesky(normal=True), ()),
    (optx.Cholesky(), optx.positive_semidefinite_tag),
]


def _has(tags, tag):
    return tag is tags or (isinstance(tags, tuple) and tag in tags)


def _transpose(operator, matrix):
    return operator.T, matrix.T


def _linearise(operator, matrix):
    return optx.linearise(operator), matrix


def _materialise(operator, matrix):
    return optx.materialise(operator), matrix


def _params():
    for make_operator in _make_operators:
        for solver, tags in _solvers:
            if make_operator is _make_diagonal_operator and not _has(
                tags, optx.diagonal_tag
            ):
                continue
            for ops in (lambda x, y: (x, y), _transpose, _linearise, _materialise):
                yield make_operator, solver, tags, ops


@pytest.mark.parametrize("make_operator,solver,tags,ops", _params())
def test_matrix_small_wellposed(make_operator, solver, tags, ops, getkey):
    if jax.config.jax_enable_x64:
        tol = 1e-10
    else:
        tol = 1e-4
    while True:
        matrix = jr.normal(getkey(), (3, 3))
        if isinstance(solver, (optx.Cholesky, optx.CG)):
            if solver.normal:
                cond_cutoff = math.sqrt(1000)
            else:
                cond_cutoff = 1000
                matrix = matrix @ matrix.T
        else:
            cond_cutoff = 1000
            if _has(tags, optx.diagonal_tag):
                matrix = jnp.diag(jnp.diag(matrix))
            if _has(tags, optx.symmetric_tag):
                matrix = matrix + matrix.T
            if _has(tags, optx.lower_triangular_tag):
                matrix = jnp.tril(matrix)
            if _has(tags, optx.upper_triangular_tag):
                matrix = jnp.triu(matrix)
            if _has(tags, optx.unit_diagonal_tag):
                matrix = matrix.at[jnp.arange(3), jnp.arange(3)].set(1)
        if jnp.linalg.cond(matrix) < cond_cutoff:
            break
    operator = make_operator(matrix, tags)
    operator, matrix = ops(operator, matrix)
    assert shaped_allclose(operator.as_matrix(), matrix, rtol=tol, atol=tol)
    true_x = jr.normal(getkey(), (3,))
    b = matrix @ true_x
    x = optx.linear_solve(operator, b, solver=solver).value
    jax_x = jnp.linalg.solve(matrix, b)
    assert shaped_allclose(x, true_x, atol=tol, rtol=tol)
    assert shaped_allclose(x, jax_x, atol=tol, rtol=tol)


def test_cg_stabilisation():
    a = jnp.array(
        [
            [0.47580394, 1.7310725, 1.4352472],
            [-0.00429837, 0.43737498, 1.006149],
            [0.00334679, -0.10773867, -0.22798078],
        ]
    )
    true_x = jnp.array([1.4491767, -1.6469518, -0.02669191])

    problem = optx.MatrixLinearOperator(a)

    solver = optx.CG(normal=True, rtol=0, atol=0, max_steps=50, stabilise_every=None)
    x = optx.linear_solve(problem, a @ true_x, solver=solver, throw=False).value
    assert jnp.all(jnp.isnan(x))
    # Likewise, this produces NaNs:
    # jsp.sparse.linalg.cg(lambda x: a.T @ (a @ x), a.T @ a @ x, tol=0)[0]

    solver = optx.CG(normal=True, rtol=0, atol=0, max_steps=50, stabilise_every=10)
    x = optx.linear_solve(problem, a @ true_x, solver=solver, throw=False).value
    assert jnp.all(jnp.invert(jnp.isnan(x)))
    assert shaped_allclose(x, true_x, rtol=1e-3, atol=1e-3)


def test_nontrivial_pytree_operator():
    x = [[1, 5.0], [jnp.array(-2), jnp.array(-2.0)]]
    y = [3, 4]
    struct = jax.eval_shape(lambda: y)
    operator = optx.PyTreeLinearOperator(x, struct)
    out = optx.linear_solve(operator, y).value
    true_out = [jnp.array(-3.25), jnp.array(1.25)]
    assert shaped_allclose(out, true_out)


def test_nonsquare_pytree_operator():
    x = [[1, 5.0, jnp.array(-1.0)], [jnp.array(-2), jnp.array(-2.0), 3]]
    y = [3, 4]
    struct = jax.eval_shape(lambda: y)
    operator = optx.PyTreeLinearOperator(x, struct)
    out = optx.linear_solve(operator, y).value
    true_out = ...  # TODO: fix this test
    assert shaped_allclose(out, true_out)
