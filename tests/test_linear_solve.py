import math

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import optimistix as optx

from .helpers import getkey, shaped_allclose


_make_operators = []


@_make_operators.append
def _make_matrix_operator(matrix, pattern):
    return optx.MatrixLinearOperator(matrix, pattern)


@_make_operators.append
def _make_trivial_pytree_operator(matrix, pattern):
    struct = jax.ShapeDtypeStruct((3,), matrix.dtype)
    return optx.PyTreeLinearOperator(matrix, struct, pattern)


@_make_operators.append
def _make_function_operator(matrix, pattern):
    fn = lambda x: matrix @ x
    in_struct = jax.ShapeDtypeStruct((3,), matrix.dtype)
    return optx.FunctionLinearOperator(fn, in_struct, pattern)


@_make_operators.append
def _make_jac_operator(matrix, pattern):
    x = jr.normal(getkey(), (3,))
    b = jr.normal(getkey(), (3, 3))
    fn_tmp = lambda x, _: x + b @ x**2
    jac = jax.jacfwd(fn_tmp)(x, None)
    diff = matrix - jac
    fn = lambda x, _: x + b @ x**2 + diff @ x
    return optx.JacobianLinearOperator(fn, x, None, pattern)


if jax.config.jax_enable_x64:
    tol = 1e-12
else:
    tol = 1e-6
_solvers = [
    (optx.AutoLinearSolver(), optx.Pattern()),
    (optx.Triangular(), optx.Pattern(lower_triangular=True)),
    (optx.Triangular(), optx.Pattern(upper_triangular=True)),
    (optx.Triangular(), optx.Pattern(lower_triangular=True, unit_diagonal=True)),
    (optx.Triangular(), optx.Pattern(upper_triangular=True, unit_diagonal=True)),
    (optx.Diagonal(), optx.Pattern(diagonal=True)),
    (optx.Diagonal(), optx.Pattern(diagonal=True, unit_diagonal=True)),
    (optx.LU(), optx.Pattern()),
    (optx.QR(), optx.Pattern()),
    (optx.SVD(), optx.Pattern()),
    (optx.CG(normal=True, materialise=True, rtol=tol, atol=tol), optx.Pattern()),
    (optx.CG(normal=True, materialise=False, rtol=tol, atol=tol), optx.Pattern()),
    (
        optx.CG(normal=False, materialise=True, rtol=tol, atol=tol),
        optx.Pattern(symmetric=True),
    ),
    (
        optx.CG(normal=False, materialise=False, rtol=tol, atol=tol),
        optx.Pattern(symmetric=True),
    ),
    (optx.Cholesky(), optx.Pattern(symmetric=True)),
    (optx.Cholesky(normal=True), optx.Pattern()),
]


@pytest.mark.parametrize("make_operator", _make_operators)
@pytest.mark.parametrize("solver,pattern", _solvers)
def test_matrix_small_wellposed(make_operator, solver, pattern, getkey):
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
            if pattern.diagonal:
                matrix = jnp.diag(jnp.diag(matrix))
            if pattern.symmetric:
                matrix = matrix + matrix.T
            if pattern.lower_triangular:
                matrix = jnp.tril(matrix)
            if pattern.upper_triangular:
                matrix = jnp.triu(matrix)
            if pattern.unit_diagonal:
                matrix = matrix.at[jnp.arange(3), jnp.arange(3)].set(1)
        if jnp.linalg.cond(matrix) < cond_cutoff:
            break
    operator = make_operator(matrix, pattern)
    assert shaped_allclose(operator.as_matrix(), matrix, rtol=tol, atol=tol)
    true_x = jr.normal(getkey(), (3,))
    b = operator.mv(true_x)
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
