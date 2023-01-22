import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import optimistix as optx

from .helpers import getkey, shaped_allclose


_make_operators = []


@_make_operators.append
def _make_matrix(matrix, pattern):
    if pattern == optx.Pattern():
        return matrix
    else:
        return None


@_make_operators.append
def _make_identity_operator(matrix, pattern):
    if pattern == optx.Pattern():
        return optx.IdentityLinearOperator(jax.ShapeDtypeStruct((3,), matrix.dtype))
    else:
        return None


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
    if pattern.symmetric:
        _pow = lambda x: x
    else:
        _pow = lambda x: x**2
    if pattern.unit_diagonal:
        off_matrix = matrix.at[jnp.arange(3), jnp.arange(3)].set(0)
        fn = lambda x, _: x + off_matrix @ _pow(x)
    else:
        fn = lambda x, _: matrix @ _pow(x)
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


def _params():
    for solver, pattern in _solvers:
        cond_cutoff = 1000
        while True:
            matrix = jr.normal(getkey(), (3, 3))
            if isinstance(solver, optx.Cholesky) and not solver.normal:
                matrix = matrix @ matrix.T
            else:
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
        for make_operator in _make_operators:
            operator = make_operator(matrix, pattern)
            if operator is not None:
                yield operator, solver


@pytest.mark.parametrize("operator,solver", _params())
def test_matrix_small_wellposed(operator, solver):
    true_x = jr.normal(getkey(), (3,))
    if isinstance(operator, optx.AbstractLinearOperator):
        matrix = operator.as_matrix()
        b = operator.mv(true_x)
    else:
        matrix = operator
        b = operator @ true_x
    x = optx.linear_solve(operator, b, solver=solver).value
    jax_x = jnp.linalg.solve(matrix, b)
    if jax.config.jax_enable_x64:
        tol = 1e-10
    else:
        tol = 1e-4
    assert shaped_allclose(x, true_x, atol=tol, rtol=tol)
    assert shaped_allclose(x, jax_x, atol=tol, rtol=tol)
