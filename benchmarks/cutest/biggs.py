import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BIGGS6(AbstractUnconstrainedMinimisation, strict=True):
    """Biggs EXP problem in 6 variables.

    Source: Problem 21 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-6-0
    """

    n: int = 6  # Problem has 6 variables
    m: int = 13  # Number of data points

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5, x6 = y

        # Define indices from 1 to m
        indices = jnp.arange(1, self.m + 1)

        # Define inner function to compute residual for a single index i
        def compute_residual(i):
            t = -0.1 * i

            # Target value calculation from the SIF file
            y_val = jnp.exp(t) - 5.0 * jnp.exp(-1.0 * i) + 3.0 * jnp.exp(4.0 * t)

            # Model calculation
            y_pred = x3 * jnp.exp(t * x1) - x4 * jnp.exp(t * x2) + x6 * jnp.exp(t * x5)

            # Return squared residual
            return (y_pred - y_val) ** 2

        # Vectorize the function over indices and sum the results
        residuals = jax.vmap(compute_residual)(indices)
        return jnp.sum(residuals)

    def y0(self):
        # Initial values from SIF file
        return jnp.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is known to be (1, 10, 1, 5, 4, 3)
        # per other references, including Buckley's original paper
        return jnp.array([1.0, 10.0, 1.0, 5.0, 4.0, 3.0])

    def expected_objective_value(self):
        # According to the SIF file comment (line 135),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
