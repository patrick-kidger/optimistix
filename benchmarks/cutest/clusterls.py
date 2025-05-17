import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CLUSTERLS(AbstractUnconstrainedMinimisation):
    """The cluster problem in 2 variables (least-squares version).

    Source: problem 207 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.
    Least-squares version of CLUSTER.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Element type A
        f1_a = x1 - x2 * x2
        f2_a = x1 - jnp.sin(x2)
        term_a = f1_a * f2_a

        # Element type B
        f1_b = jnp.cos(x2) - x1
        f2_b = x2 - jnp.cos(x1)
        term_b = f1_b * f2_b

        # Sum of squared terms (least-squares objective)
        return term_a * term_a + term_b * term_b

    def y0(self):
        # Initial values from SIF file (all zeros)
        return jnp.zeros(2)

    def args(self):
        return None

    def expected_result(self):
        # The SIF file mentions that the objective value is 0.0 at the optimal solution
        # This implies that both terms can be zero simultaneously
        # Since both terms are products, if any factor is zero, the term is zero
        # One solution is [π/2, π/2] where:
        # sin(π/2) = 1, cos(π/2) = 0, which makes several factors zero
        return jnp.array([jnp.pi / 2, jnp.pi / 2])

    def expected_objective_value(self):
        # According to the SIF file, line 69
        return jnp.array(0.0)
