import jax.random as jr

import optimistix as optx

from .helpers import shaped_allclose


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
