import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class JENSMP(AbstractUnconstrainedMinimisation, strict=True):
    """The Jennrich and Sampson problem.

    This function is a nonlinear least squares with m groups (default 10).
    Each group has two nonlinear elements.

    Source: Problem 6 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Classification: SUR2-AN-2-0
    """

    m: int = 10  # Number of groups in the least-squares problem

    def objective(self, y, args):
        """Compute the objective function value.

        For each i=1,..,m, the residual is:
            exp(i*x1) + exp(i*x2) - (2 + 2i)

        The objective is the sum of squares of these residuals.
        """
        x1, x2 = y

        # Define the residual function for a single group i
        def residual_fn(i):
            i_float = jnp.float32(i + 1)  # Convert to float and adjust for 1-indexing
            return jnp.exp(i_float * x1) + jnp.exp(i_float * x2) - (2.0 + 2.0 * i_float)

        # Compute all residuals using vmap
        residuals = jax.vmap(residual_fn)(jnp.arange(self.m))

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        """Initial point (0.3, 0.4)."""
        return jnp.array([0.3, 0.4])

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The optimal objective value is approximately 124.362."""
        return jnp.array(124.362)  # This value is mentioned in the SIF file
