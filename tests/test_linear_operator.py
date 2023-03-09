import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import optimistix as optx

from .helpers import make_diagonal_operator, make_operators, shaped_allclose


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


def _setup(matrix, tag=frozenset()):
    for make_operator in make_operators:
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
        optx.linearise(operator)


def test_materialise(getkey):
    operators = _setup(jr.normal(getkey(), (3, 3)))
    for operator in operators:
        optx.materialise(operator)


def test_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    matrix_diag = jnp.diag(matrix)
    operators = _setup(matrix)
    for operator in operators:
        assert jnp.allclose(optx.diagonal(operator), matrix_diag)


def test_is_symmetric(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    symmetric_operators = _setup(matrix.T @ matrix, optx.symmetric_tag)
    for operator in symmetric_operators:
        assert optx.is_symmetric(operator)

    not_symmetric_operators = _setup(matrix)
    _assert_except_diag(optx.is_symmetric, not_symmetric_operators, flip_cond=True)


def test_is_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    diagonal_operators = _setup(jnp.diag(jnp.diag(matrix)), optx.diagonal_tag)
    for operator in diagonal_operators:
        assert optx.is_diagonal(operator)

    not_diagonal_operators = _setup(matrix)
    _assert_except_diag(optx.is_diagonal, not_diagonal_operators, flip_cond=True)


def test_has_unit_diagonal(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_unit_diagonal = _setup(matrix)
    for operator in not_unit_diagonal:
        assert not optx.has_unit_diagonal(operator)

    matrix_unit_diag = matrix.at[jnp.arange(3), jnp.arange(3)].set(1)
    unit_diagonal = _setup(matrix_unit_diag, optx.unit_diagonal_tag)
    _assert_except_diag(optx.has_unit_diagonal, unit_diagonal, flip_cond=False)


def test_is_lower_triangular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    lower_triangular = _setup(jnp.tril(matrix), optx.lower_triangular_tag)
    for operator in lower_triangular:
        assert optx.is_lower_triangular(operator)

    not_lower_triangular = _setup(matrix)
    _assert_except_diag(optx.is_lower_triangular, not_lower_triangular, flip_cond=True)


def test_is_upper_triangular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    upper_triangular = _setup(jnp.triu(matrix), optx.upper_triangular_tag)
    for operator in upper_triangular:
        assert optx.is_upper_triangular(operator)

    not_upper_triangular = _setup(matrix)
    _assert_except_diag(optx.is_upper_triangular, not_upper_triangular, flip_cond=True)


def test_is_positive_semidefinite(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_positive_semidefinite = _setup(matrix)
    for operator in not_positive_semidefinite:
        assert not optx.is_positive_semidefinite(operator)

    positive_semidefinite = _setup(matrix.T @ matrix, optx.positive_semidefinite_tag)
    _assert_except_diag(
        optx.is_positive_semidefinite, positive_semidefinite, flip_cond=False
    )


def test_is_negative_semidefinite(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    not_negative_semidefinite = _setup(matrix)
    for operator in not_negative_semidefinite:
        assert not optx.is_negative_semidefinite(operator)

    negative_semidefinite = _setup(-matrix.T @ matrix, optx.negative_semidefinite_tag)
    _assert_except_diag(
        optx.is_negative_semidefinite, negative_semidefinite, flip_cond=False
    )


def test_is_nonsingular(getkey):
    matrix = jr.normal(getkey(), (3, 3))
    nonsingular_operators = _setup(matrix, optx.nonsingular_tag)
    _assert_except_diag(optx.is_nonsingular, nonsingular_operators, flip_cond=False)


def test_tangent_as_matrix(getkey):
    def _list_setup(matrix):
        return list(_setup(matrix))

    matrix = jr.normal(getkey(), (3, 3))
    t_matrix = jr.normal(getkey(), (3, 3))
    operators, t_operators = eqx.filter_jvp(_list_setup, (matrix,), (t_matrix,))
    for operator, t_operator in zip(operators, t_operators):
        t_operator = optx.TangentLinearOperator(operator, t_operator)
        if isinstance(operator, optx.DiagonalLinearOperator):
            assert jnp.allclose(operator.as_matrix(), jnp.diag(jnp.diag(matrix)))
            assert jnp.allclose(t_operator.as_matrix(), jnp.diag(jnp.diag(t_matrix)))
        else:
            assert jnp.allclose(operator.as_matrix(), matrix)
            assert jnp.allclose(t_operator.as_matrix(), t_matrix)
