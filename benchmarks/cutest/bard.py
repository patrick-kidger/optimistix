import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BARD(AbstractUnconstrainedMinimisation, strict=True):
    """Bard function in 3 variables.

    This function is a nonlinear least squares problem with 15 groups. Each
    group has a linear and a nonlinear element.

    Source: Problem 3 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#16.
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-3-0
    """

    n: int = 3  # Problem has 3 variables
    m: int = 15  # Number of least squares groups

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Constants from the SIF file
        constants = jnp.array(
            [
                0.14,
                0.18,
                0.22,
                0.25,
                0.29,
                0.32,
                0.35,
                0.39,
                0.37,
                0.58,
                0.73,
                0.96,
                1.34,
                2.10,
                4.39,
            ]
        )

        # Create array of indices from 1 to 15
        i_values = jnp.arange(1, self.m + 1)

        # Calculate u, v, w values for each group
        u_values = i_values

        # The SIF file splits the calculation into two parts: i=1..8 and i=9..15
        # For i=1..8, v = 16-i and w = i
        # For i=9..15, v = 16-i and w = 16-i
        v_values = 16.0 - i_values

        # Create w_values based on conditions in SIF file
        condition = i_values <= 8
        w_values = jnp.where(condition, i_values, 16.0 - i_values)

        # Calculate denominators: v*x2 + w*x3
        denominators = v_values * x2 + w_values * x3

        # Calculate each residual: y_i - (u_i / denominator_i)
        residuals = constants - (u_values / denominators)

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.array([1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        # Values from literature: approximately [0.082, 1.133, 2.344]
        return jnp.array([0.082, 1.133, 2.344])

    def expected_objective_value(self):
        # According to the SIF file comment (line 125),
        # the optimal objective value is 8.2149e-3
        return jnp.array(8.2149e-3)
