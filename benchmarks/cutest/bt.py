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


class BT4(AbstractConstrainedMinimisation):
    """The BT4 problem from the CUTEST collection of benchmark problems.

    The original problem seems to be unbounded. The contribution of x3 in the first
    constraint has been squared instead of cubed. (Modification done in SIF collection.)
    The problem is not convex.

    Note J.Haffner: --------------------------------------------------------------------
    Original reference: https://www.jstor.org/stable/2157674?seq=21, page 620.
    ------------------------------------------------------------------------------------

    SIF input: Ph. Toint, June 1993.
    Classification: QQR2-AN-3-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, _ = y
        return x1 - x2 + x2**3

    def y0(self):
        if self.y0_iD == 0:
            return jnp.array([3.1494, 1.4523, -3.6017])
        elif self.y0_iD == 1:
            return jnp.array([3.122, 1.489, -3.611])
        elif self.y0_iD == 2:
            return jnp.array([-0.94562, -2.35984, 4.30546])
        else:
            raise ValueError("Invalid start point ID. Valid iDs are 0, 1, or 2.")

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        con1 = x1**2 + x2**2 + x3**2 - 25.0
        con2 = x1 + x2 + x3 - 1.0
        return jnp.array([con1, con2]), None

    def bounds(self):
        # From BOUNDS section: all variables are free (FR)
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array([3.28903771])


class BT5(AbstractConstrainedMinimisation):
    """The BT5 problem from the CUTEST collection of benchmark problems.

    Note (from SIF file): The problem as stated in the paper seems to contain a typo.
    The sign of the x3 squared term in the first constraint has been set to [plus]
    instead of [minus] in order to ensure that the problem is bounded below and the
    optimal point stated recovered.
    The problem is not convex.

    Note J.Haffner: --------------------------------------------------------------------
    Original reference: https://www.jstor.org/stable/2157674?seq=21, page 620.
    ------------------------------------------------------------------------------------

    SIF input: Ph. Toint, June 1993.
    Classification: QQR2-AN-3-2
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0, 1, 2})

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return -1000.0 * (-(x1**2) - 2 * x2**2 - x3**2 - x1 * x2 - x1 * x3)

    def y0(self):
        if self.y0_iD == 0:
            return 2 * jnp.ones(3)
        elif self.y0_iD == 1:
            return 20 * jnp.ones(3)
        elif self.y0_iD == 2:
            return 80 * jnp.ones(3)
        else:
            raise ValueError("y0_iD must be 0, 1 or 2")

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        con1 = x1**2 + x2**2 + x3**2 - 25.0
        con2 = 8.0 * x1 + 14.0 * x2 + 7.0 * x3 - 56.0
        return jnp.array([con1, con2]), None

    def bounds(self):
        # From BOUNDS section: all variables are free (FR)
        return None

    def expected_result(self):
        # The expected result given in the original source. (Not given in SIF file.)
        # (Addition J. Haffner)
        return jnp.array([3.5121, 0.21699, 3.5522])

    def expected_objective_value(self):
        # From OBJECT BOUND section of SIF file, LO SOLTN value
        return jnp.array(961.71517219)
