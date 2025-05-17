import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO needs human review
class GAUSSIAN(AbstractUnconstrainedMinimisation, strict=True):
    """The GAUSSIAN function.

    More's gaussian problem in 3 variables. This is a nonlinear least-squares
    version of problem ARGAUSS.

    Source: Problem 9 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#28
    SIF input: Ph. Toint, Dec 1989.

    Classification SUR2-AN-3-0

    This function is a nonlinear least squares with 15 groups. Each
    group has a nonlinear element.
    """

    # Number of variables
    n: int = 3

    def objective(self, y, args):
        """Compute the objective function value.

        Args:
            y: The parameters [x1, x2, x3]
            args: None

        Returns:
            The sum of squared residuals.
        """
        del args
        x1, x2, x3 = y

        # Data points from the SIF file constants
        y_data = jnp.array(
            [
                0.0009,
                0.0044,
                0.0175,
                0.0540,
                0.1295,
                0.2420,
                0.3521,
                0.3989,
                0.3521,
                0.2420,
                0.1295,
                0.0540,
                0.0175,
                0.0044,
                0.0009,
            ]
        )

        # Create time points (t values)
        # According to the SIF file, ti = (8 - i) * 0.5 where i goes from 1 to 15
        t = jnp.array([(8 - i) * 0.5 for i in range(1, 16)])

        # Compute the model: v1 * exp(v2 * (t - v3)^2 * -0.5)
        # This comes from the SIF file definition of the Gaussian function
        diffs = t - x3
        squared_diffs = -0.5 * diffs**2
        model = x1 * jnp.exp(x2 * squared_diffs)

        # Compute residuals and sum of squared residuals
        residuals = model - y_data
        return jnp.sum(residuals**2)

    def y0(self):
        """Return the starting point from the SIF file."""
        # Initial point from the SIF file: x1 = 0.4, x2 = 1.0, x3 = 0.0
        return jnp.array([0.4, 1.0, 0.0])

    def args(self):
        """Return None as no additional args are needed."""
        return None

    def expected_result(self):
        """Return None as the exact solution is not specified in the SIF file."""
        # The SIF file doesn't specify the optimal solution
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the minimum objective value
        return None
