import equinox as eqx
import jax.numpy as jnp


# TODO: this problem is parsed and has not been verified by a human yet.
class FLETCHER(eqx.Module):
    """The FLETCHER problem from the CUTEST collection of benchmark problems.

    A problem found in Fletcher's book. The problem is to find the least area
    of a right-angled triangle which contains a circle of unit radius.

    Source: R. Fletcher, "Practical Methods of Optimization",
    second edition, Wiley, 1987.

    Classification: QOR2-AN-4-4
    """

    def objective(self, y, args):
        x1, x2, x3, x4 = y
        # From GROUP USES section and ELEMENT TYPE/USES:
        # OBJ = OBE1 = X1 * X2 (product of first two variables)
        return x1 * x2

    def y0(self):
        # From START POINT section
        return jnp.array([1.0, 1.0, 1.0, 1.0])

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4 = y
        # From GROUPS, GROUP USES, ELEMENT USES and CONSTANTS sections:
        # C1 = SPEC(x1,x2,x3,x4) - x3^2 - x4^2 - 1.0
        # The SPEC element is complex, but represents the function for a
        # circle of unit radius in the triangle. Based on the mathematical formulation
        n = x1 * x3 + x2 * x4
        d = x1**2 + x2**2
        c1_spec = (n**2) / d  # This is FV in the SPEC element

        c1 = c1_spec - x3**2 - x4**2 - 1.0

        # C2 = X1 - X3 + 1.0 >= 0
        c2 = x1 - x3 + 1.0

        # C3 = X2 - X4 + 1.0 >= 0
        c3 = x2 - x4 + 1.0

        # C4 = X3 - X4 + 0.0 >= 0 (constant is not specified, using 0.0)
        c4 = x3 - x4

        # Equality constraint c1, inequality constraints c2, c3, c4
        return jnp.array([c1]), jnp.array([c2, c3, c4])

    def bounds(self):
        # From BOUNDS section: X4 >= 1.0, others are free
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, 1.0])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper

    def expected_result(self):
        # The optimal point coordinates are not provided in the file
        return None

    def expected_objective_value(self):
        # Not provided in the file
        return None
