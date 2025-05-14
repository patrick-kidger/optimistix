import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Needs verification against another CUTEst interface
class ENGVAL1(AbstractUnconstrainedMinimisation, strict=True):
    """The ENGVAL1 function.

    This problem is a sum of 2n-2 groups, n-1 of which contain 2 nonlinear elements.

    Source: problem 31 in
    Ph.L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    See also Buckley#172 (p. 52)
    SIF input: Ph. Toint and N. Gould, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    n: int = 2  # Other dimensions suggested: 50, 100, 1000, 5000

    def objective(self, y, args):
        del args
        y2 = y**2
        nonlinear = jnp.sum((y2[:-1] + y2[1:]) ** 2)
        linear = jnp.sum(-4 * y[:-1] - 3)

        return nonlinear + linear

    def y0(self):
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)


# TODO: Needs verification against another CUTEst interface
class ENGVAL2(AbstractUnconstrainedMinimisation, strict=True):
    """The ENGVAL2 problem.

    Source: problem 15 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-3-0
    """

    def objective(self, y, args):
        del args

        x1, x2, x3 = y

        # Group G1: (x1^2 + x2^2 + x3^2)^2
        g1 = (x1**2 + x2**2 + x3**2) ** 2

        # Group G2: (x1^2 + x2^2 + (x3-2)^2)^2
        g2 = (x1**2 + x2**2 + (x3 - 2) ** 2) ** 2

        # Group G3: (x1 + x2 + x3)^2
        g3 = (x1 + x2 + x3) ** 2

        # Group G4: (x1 + x2 - x3)^2
        g4 = (x1 + x2 - x3) ** 2

        # Group G5: 36 * (3*x2^2 + (x1^3 + (5*x3-x1+1)^2)^2)
        elt_term = (5 * x3 - x1 + 1) ** 2 + x1**3
        g5 = 36 * (3 * x2**2 + elt_term**2)

        # Combining all groups with their coefficients
        return g1 + g2 + g3 - g4 + g5

    def y0(self):
        # Starting point from the file
        return jnp.array([1.0, 2.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        return None  # The file doesn't provide the solution

    def expected_objective_value(self):
        return jnp.array(0.0)  # From OBJECT BOUND in the file
