import jax.numpy as jnp

from .problem import AbstractConstrainedMinimisation


class BT2(AbstractConstrainedMinimisation):
    """The BT2 problem from the CUTEST collection of benchmark problems.

    Source: problem 2 in
    P.T. Boggs and J.W. Tolle,
    "A strategy for global convergence in a sequential quadratic programming algorithm",
    SINUM 26(3), pp. 600-623, 1989.
    Also available at: https://www.jstor.org/stable/2157674?seq=21, page 620.

    Note J.Haffner: --------------------------------------------------------------------
    I think this problem tests for (slow) convergence with objective and constraint
    functions that have locally similar shapes, but are shifted against each other.
    ------------------------------------------------------------------------------------

    SIF input: Ph. Toint, June 1993.
    Classification QQR2-AY-3-1
    """

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return (x1 - 1) ** 2 + (x1 - x2) ** 2 + (x2 - x3) ** 4

    def y0(self):
        return jnp.array([10.0, 10.0, 10.0])

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        return x1 * (1 + x2**2) + x3**4 - 4 - 3 * jnp.sqrt(2), None

    def bounds(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array([0.032568200])
