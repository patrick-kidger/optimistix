import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BRKMCC(AbstractUnconstrainedMinimisation, strict=True):
    """BRKMCC function.

    Source: Problem 85 (p.35) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-2-0
    """

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # From SIF file:
        # G1 = (x1 - 2.0)^2
        g1 = (x1 - 2.0) ** 2

        # G2 = (x2 - 1.0)^2
        g2 = (x2 - 1.0) ** 2

        # G3 = 25.0 * (1.0 / (-0.25*x1 - x2))
        g3 = 25.0 / (-0.25 * x1 - x2)

        # G4 = 0.2 * ((x1 - 2.0*x2) - 1.0)^2
        g4 = 0.2 * ((x1 - 2.0 * x2) - 1.0) ** 2

        return g1 + g2 + g3 + g4

    def y0(self):
        # Initial values from SIF file
        return jnp.array([2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        # According to literature, the minimum is around (0.91, 0.61)
        return jnp.array([0.91, 0.61])

    def expected_objective_value(self):
        # According to the SIF file comment (line 77),
        # the optimal objective value is 0.16904
        return jnp.array(0.16904)
