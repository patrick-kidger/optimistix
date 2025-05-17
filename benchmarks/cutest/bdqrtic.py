import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BDQRTIC(AbstractUnconstrainedMinimisation, strict=True):
    """BDQRTIC function.

    This problem is quartic and has a banded Hessian with bandwidth = 9.

    Source: Problem 61 in
    A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
    "Performance of a multifrontal scheme for partially separable optimization",
    Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    def __init__(self, n=100):
        """Initialize the BDQRTIC problem.

        Args:
            n: Dimension of the problem. In the original paper, values of
               100, 500, 1000, and 5000 were used. Default is 100.
        """
        self.n = n

    def objective(self, y, args):
        del args
        n = self.n

        # Sum of squares for groups 1 to n-4
        residuals = []

        for i in range(n - 4):
            # From the SIF file, each group has contributions from:
            # x[i], x[i+1], x[i+2], x[i+3], and x[n-1]
            # with coefficients 1.0, 2.0, 3.0, 4.0, and 5.0 respectively
            term1 = (
                y[i] ** 2
                + 2 * y[i + 1]
                + 3 * y[i + 2]
                + 4 * y[i + 3]
                + 5 * y[n - 1]
                - 3.0
            )

            # The term (x[i] - 3.0) mentioned in constants section (line 54)
            term2 = y[i] - 4.0

            residuals.append(term1**2 + term2**2)

        return jnp.sum(jnp.array(residuals))

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Based on the SIF file comment for n=100 (line 103)
        if self.n == 100:
            return jnp.array(3.78769e02)
        elif self.n == 500:
            return jnp.array(1.98101e03)
        elif self.n == 1000:
            return jnp.array(3.98382e03)
        else:
            # For other values of n, the optimal value is not provided
            return None
