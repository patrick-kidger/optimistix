import jax.random as jr

import optimistix as optx

from .helpers import shaped_allclose


def test_matrix_small_wellposed(getkey):
    a = jr.normal(getkey(), (3, 3))
    true_x = jr.normal(getkey(), (3,))
    b = a @ true_x
    x = optx.linear_solve(a, b).value
    assert shaped_allclose(x, true_x)
