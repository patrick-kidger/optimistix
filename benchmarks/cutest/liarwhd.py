import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class LIARWHD(AbstractUnconstrainedMinimisation, strict=True):
    """Simplified version of the NONDIA problem.

    Source:
    G. Li,
    "The secant/finite difference algorithm for solving sparse
    nonlinear systems of equations",
    SIAM Journal on Optimization, (to appear), 1990.

    SIF input: Ph. Toint, Aug 1990.
    Classification: SUR2-AN-V-0
    """

    # Problem dimension
    n: int = 5000  # Other suggested values: 36, 100, 500, 1000, 10000

    def objective(self, y, args):
        """Compute the objective function value.

        The objective consists of the sum of squared terms of the form:
        [0.25 * (-x_1 + x_i^2)]^2 for i = 1, 2, ..., n
        """
        # Calculate the terms: 0.25 * (-x_1 + x_i^2)
        x1 = y[0]
        terms = 0.25 * (-x1 + y**2)

        # Return the sum of squared terms
        return jnp.sum(terms**2)

    def y0(self):
        """Initial point with all variables set to 4.0."""
        return jnp.ones(self.n) * 4.0

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is all ones according to the problem definition."""
        return jnp.ones(self.n)

    def expected_objective_value(self):
        """The solution value is 0.0."""
        return jnp.array(0.0)
