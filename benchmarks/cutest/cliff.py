import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CLIFF(AbstractUnconstrainedMinimisation):
    """The 'cliff problem' in 2 variables.

    This is a challenging function with a steep cliff along one side.
    The objective function combines quadratic and exponential terms.

    Source: problem 206 (p. 46) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # The objective has three terms:
        # 1. 0.01*x1^2 - 0.03*0.01*x1 = 0.01*x1*(x1 - 0.03)
        # 2. x1 - x2 (This term doesn't appear as a separate group in the SIF file)
        # 3. exp(20*(x1 - x2))

        term1 = 0.01 * x1 * (x1 - 0.03)
        term2 = x1 - x2
        term3 = jnp.exp(20.0 * (x1 - x2))

        return term1 + term2 + term3

    def y0(self):
        # Initial values from SIF file
        return jnp.array([0.0, -1.0])

    def args(self):
        return None

    def expected_result(self):
        # The SIF file doesn't fully specify the optimal solution,
        # but based on the objective value we can identify the
        # approximate location of the solution
        return jnp.array([0.02, 0.0])

    def expected_objective_value(self):
        # According to the SIF file, line 62
        return jnp.array(0.199786613)
