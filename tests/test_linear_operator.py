import jax.numpy as jnp
import jax.random as jr

import optimistix as optx

from .helpers import (
    make_diagonal_operator,
    make_function_operator,
    make_jac_operator,
    make_matrix_operator,
    make_trivial_pytree_operator,
    shaped_allclose,
)


def test_ops(getkey):
    matrix1 = optx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    matrix2 = optx.MatrixLinearOperator(jr.normal(getkey(), (3, 3)))
    scalar = jr.normal(getkey(), ())
    add = matrix1 + matrix2
    composed = matrix1 @ matrix2
    mul = matrix1 * scalar
    rmul = scalar * matrix1
    div = matrix1 / scalar
    vec = jr.normal(getkey(), (3,))

    assert shaped_allclose(matrix1.mv(vec) + matrix2.mv(vec), add.mv(vec))
    assert shaped_allclose(matrix1.mv(matrix2.mv(vec)), composed.mv(vec))
    scalar_matvec = scalar * matrix1.mv(vec)
    assert shaped_allclose(scalar_matvec, mul.mv(vec))
    assert shaped_allclose(scalar_matvec, rmul.mv(vec))
    assert shaped_allclose(matrix1.mv(vec) / scalar, div.mv(vec))

    add_matrix = matrix1.as_matrix() + matrix2.as_matrix()
    composed_matrix = matrix1.as_matrix() @ matrix2.as_matrix()
    mul_matrix = scalar * matrix1.as_matrix()
    div_matrix = matrix1.as_matrix() / scalar
    assert shaped_allclose(add_matrix, add.as_matrix())
    assert shaped_allclose(composed_matrix, composed.as_matrix())
    assert shaped_allclose(mul_matrix, mul.as_matrix())
    assert shaped_allclose(mul_matrix, rmul.as_matrix())
    assert shaped_allclose(div_matrix, div.as_matrix())

    assert shaped_allclose(add_matrix.T, add.T.as_matrix())
    assert shaped_allclose(composed_matrix.T, composed.T.as_matrix())
    assert shaped_allclose(mul_matrix.T, mul.T.as_matrix())
    assert shaped_allclose(mul_matrix.T, rmul.T.as_matrix())
    assert shaped_allclose(div_matrix.T, div.T.as_matrix())


_create_operator_types = [
    make_matrix_operator,
    make_trivial_pytree_operator,
    make_function_operator,
    make_jac_operator,
    make_diagonal_operator,
]


def _setup(matrix, tag=frozenset()):
    for make_operator in _create_operator_types:
        if make_operator is make_diagonal_operator:
            tag2 = optx.diagonal_tag
        else:
            tag2 = tag
        operator = make_operator(matrix, tag2)
    yield operator


def _assert_except_diag(cond_fun, operators, flip_cond):
    if flip_cond:
        _cond_fun = cond_fun
        cond_fun = lambda x: not _cond_fun(x)
    for operator in operators:
        if isinstance(operator, optx.DiagonalLinearOperator):
            assert not cond_fun(operator)
        else:
            assert cond_fun(operator)


def test_linearise(getkey):
    operators = _setup(jr.normal(getkey(), (3, 3)))

    for operator in operators:
        if isinstance(operator, optx.JacobianLinearOperator):
            optx.linearise(operator)
        else:
            assert optx.linearise(operator) == operator


def test_materialise(getkey):
    operators = _setup(jr.normal(getkey(), (3, 3)))

    for operator in operators:
        if isinstance(
            operator, (optx.FunctionLinearOperator, optx.JacobianLinearOperator)
        ):
            optx.materialise(operator)
        else:
            assert optx.materialise(operator) == operator


def test_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    matrix_diag = jnp.diag(matrix)
    operators = _setup(matrix)

    for operator in operators:
        assert jnp.allclose(optx.diagonal(operator), matrix_diag)


def test_is_symmetric(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_symmetric_operators = _setup(matrix)
    symmetric_operators = _setup(matrix.T @ matrix, optx.symmetric_tag)

    _assert_except_diag(optx.is_symmetric, not_symmetric_operators, True)

    for operator in symmetric_operators:
        assert optx.is_symmetric(operator)


def test_is_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_diagonal_operators = _setup(matrix)
    diagonal_operators = _setup(jnp.diag(jnp.diag(matrix)), optx.diagonal_tag)

    _assert_except_diag(optx.is_diagonal, not_diagonal_operators, True)

    for operator in diagonal_operators:
        assert optx.is_diagonal(operator)


def test_has_unit_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_unit_diagonal = _setup(matrix)
    unit_diag = _setup(
        matrix.at[jnp.arange(3), jnp.arange(3)].set(1), optx.unit_diagonal_tag
    )

    for operator in not_unit_diagonal:
        assert not optx.has_unit_diagonal(operator)

    _assert_except_diag(optx.has_unit_diagonal, unit_diag, flip_cond=False)


def test_is_lower_triangular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_lower_triangular = _setup(matrix)
    lower_triangular = _setup(jnp.tril(matrix), optx.lower_triangular_tag)

    _assert_except_diag(optx.is_lower_triangular, not_lower_triangular, flip_cond=True)

    for operator in lower_triangular:
        assert optx.is_lower_triangular(operator)


def test_is_upper_triangular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_upper_triangular = _setup(matrix)
    upper_triangular = _setup(jnp.triu(matrix), optx.upper_triangular_tag)

    _assert_except_diag(optx.is_upper_triangular, not_upper_triangular, flip_cond=True)

    for operator in upper_triangular:
        assert optx.is_upper_triangular(operator)


def test_is_positive_semidefinite(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_positive_semidefinite = _setup(matrix)
    positive_semidefinite = _setup(matrix.T @ matrix, optx.positive_semidefinite_tag)

    for operator in not_positive_semidefinite:
        assert not optx.is_positive_semidefinite(operator)

    _assert_except_diag(
        optx.is_positive_semidefinite, positive_semidefinite, flip_cond=False
    )


def test_is_negative_semidefinite(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_negative_semidefinite = _setup(matrix)
    negative_semidefinite = _setup(-matrix.T @ matrix, optx.negative_semidefinite_tag)

    for operator in not_negative_semidefinite:
        assert not optx.is_negative_semidefinite(operator)

    _assert_except_diag(
        optx.is_negative_semidefinite, negative_semidefinite, flip_cond=False
    )


def test_is_nonsingular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    nonsingular_operators = _setup(matrix, optx.nonsingular_tag)

    _assert_except_diag(optx.is_nonsingular, nonsingular_operators, flip_cond=False)
