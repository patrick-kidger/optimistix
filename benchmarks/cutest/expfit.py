import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class EXPFIT(AbstractUnconstrainedMinimisation):
    """A simple exponential fit problem.

    This problem involves fitting an exponential function of the form
    f(x) = ALPHA * exp(BETA * x) to 10 data points.

    Source: Problem 8 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-2-0
    """

    def objective(self, y, args):
        del args

        # Extract variables
        alpha, beta = y

        # Define the data points
        # x values: i * 0.25 for i = 1,...,10
        x_values = jnp.arange(1, 11) * 0.25

        # Define a function to compute a single residual given an x value
        def compute_residual(x):
            # Model: alpha * exp(beta * x)
            model_value = alpha * jnp.exp(beta * x)

            # Target values aren't explicitly given in the SIF file
            # The objective is the sum of squared model values
            return model_value

        # Compute residuals for all data points
        residuals = jax.vmap(compute_residual)(x_values)

        # Sum of squared residuals
        return jnp.sum(jnp.square(residuals))

    def y0(self):
        # Starting point from the SIF file: alpha = 2.5, beta = 0.25
        return jnp.array([2.5, 0.25])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not specified in the SIF file
        return None

    def expected_objective_value(self):
        return jnp.array(8.7945855171)
