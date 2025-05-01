import jax.numpy as jnp

from .problem import AbstractConstrainedMinimisation


class BT1(AbstractConstrainedMinimisation):
    """The BT1 problem from the CUTEST collection of benchmark problems.

    Note J.Haffner: --------------------------------------------------------------------
    Original reference: https://www.jstor.org/stable/2157674?seq=21, page 620.

    This problem consists of finding the minimum along a circular line. A multiple of
    100 of the constraint violation is added to the objective function, which makes an
    otherwise trivial problem artificially difficult - initially the objective function
    is dominated by the constraint violation, and progress towards the minimum can be
    impeded by small deviations from the circular line, leading to very small steps
    taken by most solvers and consequently slow convergence.
    This problem is also sensitive to poor initialisation of the dual variables.
    ------------------------------------------------------------------------------------

    Source: problem 1 in P.T. Boggs and J.W. Tolle,
    "A strategy for global convergence in a sequential quadratic programming algorithm",
    SINUM 26(3), pp. 600-623, 1989.

    SIF input: Ph. Toint, June 1993.
    Classification QQR2-AN-2-1
    """

    def objective(self, y, args):
        del args
        x1, x2 = y
        # Constant factor was changed from original, where it was 10, to 100 in CUTEST
        return -x1 + 100 * (x1**2 + x2**2 - 1)

    def y0(self):
        return jnp.array([0.08, 0.06])

    def args(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        return x1**2 + x2**2 - 1, None

    def bounds(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0])  # Analytical solution (J. Haffner)

    def expected_objective_value(self):
        return jnp.array(-1.0)


class BT2(AbstractConstrainedMinimisation):
    """The BT2 problem from the CUTEST collection of benchmark problems.

    Source: problem 2 in
    P.T. Boggs and J.W. Tolle,
    "A strategy for global convergence in a sequential quadratic programming algorithm",
    SINUM 26(3), pp. 600-623, 1989.

    Note J.Haffner: --------------------------------------------------------------------
    Original reference: https://www.jstor.org/stable/2157674?seq=21, page 620.

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
