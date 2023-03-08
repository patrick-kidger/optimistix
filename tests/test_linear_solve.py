import math

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import optimistix as optx

from .helpers import has_tag, make_diagonal_operator, make_operators, shaped_allclose


jax.config.update("jax_enable_x64", True)

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
    (optx.CG(normal=True, rtol=tol, atol=tol), optx.negative_semidefinite_tag),
    (optx.CG(normal=False, rtol=tol, atol=tol), optx.negative_semidefinite_tag),
    (optx.Cholesky(), optx.positive_semidefinite_tag),
    (optx.Cholesky(), optx.negative_semidefinite_tag),
]


def _transpose(operator, matrix):
    return operator.T, matrix.T


def _linearise(operator, matrix):
    return optx.linearise(operator), matrix


def _materialise(operator, matrix):
    return optx.materialise(operator), matrix


def _params():
    for make_operator in make_operators:
        for solver, tags in _solvers:
            if make_operator is make_diagonal_operator and not has_tag(
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
        if isinstance(solver, optx.CG):
            if solver.normal:
                cond_cutoff = math.sqrt(1000)
            else:
                cond_cutoff = 1000
                matrix = matrix @ matrix.T
        elif isinstance(solver, optx.Cholesky):
            cond_cutoff = 1000
            matrix = matrix @ matrix.T
            if has_tag(tags, optx.negative_semidefinite_tag):
                matrix = -matrix
        else:
            cond_cutoff = 1000
            if has_tag(tags, optx.diagonal_tag):
                matrix = jnp.diag(jnp.diag(matrix))
            if has_tag(tags, optx.symmetric_tag):
                matrix = matrix + matrix.T
            if has_tag(tags, optx.lower_triangular_tag):
                matrix = jnp.tril(matrix)
            if has_tag(tags, optx.upper_triangular_tag):
                matrix = jnp.triu(matrix)
            if has_tag(tags, optx.unit_diagonal_tag):
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
        ],
        dtype=jnp.float32,
    )
    true_x = jnp.array([1.4491767, -1.6469518, -0.02669191], dtype=jnp.float32)

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


@pytest.mark.parametrize("solver", (optx.LU(), optx.QR(), optx.SVD()))
def test_mixed_dtypes(solver):
    f32 = lambda x: jnp.array(x, dtype=jnp.float32)
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    x = [[f32(1), f64(5)], [f32(-2), f64(-2)]]
    y = [f64(3), f64(4)]
    struct = jax.eval_shape(lambda: y)
    operator = optx.PyTreeLinearOperator(x, struct)
    out = optx.linear_solve(operator, y, solver=solver).value
    true_out = [f32(-3.25), f64(1.25)]
    assert shaped_allclose(out, true_out)


def test_mixed_dtypes_triangular():
    f32 = lambda x: jnp.array(x, dtype=jnp.float32)
    f64 = lambda x: jnp.array(x, dtype=jnp.float64)
    x = [[f32(1), f64(0)], [f32(-2), f64(-2)]]
    y = [f64(3), f64(4)]
    struct = jax.eval_shape(lambda: y)
    operator = optx.PyTreeLinearOperator(x, struct, optx.lower_triangular_tag)
    out = optx.linear_solve(operator, y, solver=optx.Triangular()).value
    true_out = [f32(3), f64(-5)]
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize("solver", (optx.AutoLinearSolver(), optx.QR(), optx.SVD()))
def test_nonsquare_pytree_operator1(solver):
    x = [[1, 5.0, jnp.array(-1.0)], [jnp.array(-2), jnp.array(-2.0), 3.0]]
    y = [3.0, 4]
    struct = jax.eval_shape(lambda: y)
    operator = optx.PyTreeLinearOperator(x, struct)
    out = optx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, 5.0, -1.0], [-2.0, -2.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))
    true_out = [true_out[0], true_out[1], true_out[2]]
    assert shaped_allclose(out, true_out)


@pytest.mark.parametrize("solver", (optx.AutoLinearSolver(), optx.QR(), optx.SVD()))
def test_nonsquare_pytree_operator2(solver):
    x = [[1, jnp.array(-2)], [5.0, jnp.array(-2.0)], [jnp.array(-1.0), 3.0]]
    y = [3.0, 4, 5.0]
    struct = jax.eval_shape(lambda: y)
    operator = optx.PyTreeLinearOperator(x, struct)
    out = optx.linear_solve(operator, y, solver=solver).value
    matrix = jnp.array([[1.0, -2.0], [5.0, -2.0], [-1.0, 3.0]])
    true_out, _, _, _ = jnp.linalg.lstsq(matrix, jnp.array(y))
    true_out = [true_out[0], true_out[1]]
    assert shaped_allclose(out, true_out)
