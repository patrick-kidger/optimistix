import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class KSSLS(AbstractUnconstrainedMinimisation, strict=True):
    """KSS system with a zero root having exponential multiplicity by dimension.

    This is a least-squares version of KSS.

    Source: problem 8.1 in
    Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
     at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics
    University of Notre Dame, Indiana, USA (2012)

    SIF input: Nick Gould, Jan 2012.
    Least-squares version of KSS.SIF, Nick Gould, Jan 2020.
    Classification: SUR2-AN-V-0
    """

    # Problem dimension
    n: int = 1000  # Other suggested values: 4, 10, 100

    def objective(self, y, args):
        """Compute the objective function value.

        The system is composed of n equations, where each equation i is:
        sum_{j=1, j!=i}^n x_j - 3*x_i = -1

        For each equation, we compute the squared residual and then sum them.
        """
        n = self.n

        # Define the residual function for a single equation i
        def residual_fn(i):
            # Sum all variables except x_i
            other_sum = jnp.sum(y) - y[i]
            # Calculate residual: sum_{j=1, j!=i}^n x_j - 3*x_i + 1
            return other_sum - 3.0 * y[i] + 1.0

        # Compute residuals for all equations using vmap
        indices = jnp.arange(n)
        residuals = jax.vmap(residual_fn)(indices)

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        """Initial point with all variables set to 1000."""
        return jnp.ones(self.n) * 1000.0

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is zero for all variables."""
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        """The solution value mentioned in the SIF file comment."""
        return jnp.array(0.0)
