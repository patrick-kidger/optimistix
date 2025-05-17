import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class GULF(AbstractUnconstrainedMinimisation, strict=True):
    """The Gulf RD test problem in 3 variables.

    This function is a nonlinear least squares with 99 groups. Each
    group has a nonlinear element of exponential type.

    Source: problem 11 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#27
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-MN-3-0
    """

    n: int = 3  # Number of variables is 3
    m: int = 99  # Number of data points

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Create array of t_i values: t_i = 0.01 * i for i=1...m
        t_values = jnp.arange(1, self.m + 1) * 0.01

        # Compute the nonlinear expressions as defined in the SIF file (lines 133-142)
        # y_i terms: 25 + (-50 * log(t_i))^(2/3)
        y_i = 25.0 + jnp.power(-50.0 * jnp.log(t_values), 2.0 / 3.0)

        # y_i - x2 term
        y_minus_v2 = y_i - x2

        # Calculate |y_i - v_2|^v_3 / v_1
        a_i = jnp.power(jnp.abs(y_minus_v2), x3) / x1

        # Compute exp(-a_i) which are the residuals for each group
        residuals = jnp.exp(-a_i)

        # Sum of squares of residuals (as per the L2 group type in SIF file)
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file
        return jnp.array([5.0, 2.5, 0.15])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to SIF file (line 100), the optimal objective value is 0.0
        return jnp.array(0.0)
