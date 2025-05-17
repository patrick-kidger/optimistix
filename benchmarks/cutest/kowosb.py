import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class KOWOSB(AbstractUnconstrainedMinimisation, strict=True):
    """A problem arising in the analysis of kinetic data for an enzyme reaction.

    Known as the Kowalik and Osborne problem in 4 variables.
    This function is a nonlinear least squares with 11 groups.
    Each group has a linear and a nonlinear element.

    Source: Problem 15 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Classification: SUR2-MN-4-0
    """

    # Observed response values
    y_values: jnp.ndarray = jnp.array(
        [
            0.1957,
            0.1947,
            0.1735,
            0.1600,
            0.0844,
            0.0627,
            0.0456,
            0.0342,
            0.0323,
            0.0235,
            0.0246,
        ]
    )

    # Independent variable values (u)
    u_values: jnp.ndarray = jnp.array(
        [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0624]
    )

    def model(self, u, params):
        """Compute the model function: (v1 * (u^2 + u*v2)) / (u^2 + u*v3 + v4)"""
        v1, v2, v3, v4 = params
        numerator = v1 * (u**2 + u * v2)
        denominator = u**2 + u * v3 + v4
        return numerator / denominator

    def objective(self, y, args):
        """Compute the objective function value.

        For each data point i, the residual is:
        model(u_i, [v1, v2, v3, v4]) - y_i

        The objective is the sum of squares of these residuals.
        """
        # Calculate the predicted values using the model
        y_pred = jax.vmap(lambda u: self.model(u, y))(self.u_values)

        # Calculate the residuals
        residuals = y_pred - self.y_values

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        """Initial point from the SIF file."""
        return jnp.array([0.25, 0.39, 0.415, 0.39])

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The solution value mentioned in the SIF file comment."""
        return jnp.array(0.00102734)
