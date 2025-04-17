import equinox as eqx
import jax.numpy as jnp


class FLT(eqx.Module):
    """The FLT problem from the CUTEST collection of benchmark problems.

    Source: A troublesome problem for filter methods
    R. Fletcher, S. Leyffer and Ph. L. Toint,
    "On the global convergence of a filter-SQP method",
    SIAM J. Optimization 13 2002:44-59.

    SIF input: Nick Gould, May 2008
    Classification: QOR2-AN-2-2
    """

    def objective(self, y, args):
        del args
        _, x2 = y
        # From GROUPS section and GROUP USES/TYPE:
        # OBJ = L2(X2 + 1.0) = (X2 + 1.0)^2
        return (x2 + 1.0) ** 2

    def y0(self):
        # From START POINT section
        return jnp.array([1.0, 0.0])

    def args(self):
        return None

    def constraint(self, y):
        x1, _ = y
        # From GROUP USES section and ELEMENT TYPE/USES:
        # CON1 = E1 = SQ(X1) = X1^2
        # CON2 = E2 = CUBE(X1) = X1^3
        con1 = x1**2
        con2 = x1**3
        return jnp.array([con1, con2]), None

    def bounds(self):
        # From BOUNDS section: XR indicates both variables are "real", so no bounds
        return None

    def expected_result(self):
        # The optimal point coordinates are not provided in the SIF file, but analysis
        # of the problem reveals that the optimal point is (0, -1). At this point, we
        # recover the expected objective value (which is given in the SIF file.)
        # Both constraints are zero if x1 = 0.0, and the parabolic objective is zero if
        # x2 = -1.0. (jh)
        return jnp.array([0.0, -1.0])

    def expected_objective_value(self):
        # From OBJECT BOUND section, LO SOLTN value
        return 0.0
