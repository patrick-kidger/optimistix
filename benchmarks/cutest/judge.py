import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class JUDGE(AbstractUnconstrainedMinimisation, strict=True):
    """SCIPY global optimization benchmark example Judge.

    Fit: y = x_1 + a_i * x_2 + b_i^2 * x_2 + e

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020
    Classification: SUR2-MN-2-0
    """

    def __init__(self):
        # Data values from the problem definition
        self.a_values = jnp.array(
            [
                0.286,
                0.973,
                0.384,
                0.276,
                0.973,
                0.543,
                0.957,
                0.948,
                0.543,
                0.797,
                0.936,
                0.889,
                0.006,
                0.828,
                0.399,
                0.617,
                0.939,
                0.784,
                0.072,
                0.889,
            ]
        )

        self.b_values = jnp.array(
            [
                0.645,
                0.585,
                0.310,
                0.058,
                0.455,
                0.779,
                0.259,
                0.202,
                0.028,
                0.099,
                0.142,
                0.296,
                0.175,
                0.180,
                0.842,
                0.039,
                0.103,
                0.620,
                0.158,
                0.704,
            ]
        )

        self.y_values = jnp.array(
            [
                4.284,
                4.149,
                3.877,
                0.533,
                2.211,
                2.389,
                2.145,
                3.231,
                1.998,
                1.379,
                2.106,
                1.428,
                1.011,
                2.179,
                2.858,
                1.388,
                1.651,
                1.593,
                1.046,
                2.152,
            ]
        )

    def objective(self, y, args):
        """Compute the objective function value.

        For each data point i, the residual is:
        x_1 + a_i * x_2 + (b_i^2) * x_2 - y_i

        The objective is the sum of squares of these residuals.
        """
        x1, x2 = y

        # Calculate the predicted values using the model
        # y_pred = x_1 + a_i * x_2 + (b_i^2) * x_2
        b_squared = self.b_values**2
        y_pred = x1 + (self.a_values + b_squared) * x2

        # Calculate the residuals
        residuals = y_pred - self.y_values

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        """Initial point (1.0, 5.0)."""
        return jnp.array([1.0, 5.0])

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The solution value seems to be 0.0 according to the SIF file comment."""
        return jnp.array(0.0)
