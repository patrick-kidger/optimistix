import equinox as eqx
import jax.numpy as jnp


class BT5(eqx.Module):
    """The BT5 problem from the CUTEST collection of benchmark problems.
    Originally described in: https://www.jstor.org/stable/2157674?seq=21, on page 620.

    Note (from SIF file): The problem as stated in the paper seems to contain a typo.
    The sign of the x3 squared term in the first constraint has been set to [plus]
    instead of [minus] in order to ensure that the problem is bounded below and the
    optimal point stated recovered. (Described in BT5 file found at
    https://bitbucket.org/optrove/sif.)

    The problem is not convex.

    SIF input: Ph. Toint, June 1993.
    Classification: QQR2-AN-3-2
    """

    y0_iD: int = 0

    def __check_init__(self):
        assert self.y0_iD in (0, 1, 2), "y0_iD must be 0, 1 or 2."

    def objective(self, y, args):
        del args
        x1, x2, x3 = y
        return (
            1000.0 - x1**2 - 2 * x2**2 - x3**2 - x1 * x2 - x1 * x3
        )  # As in original source

    def y0(self):
        if self.y0_iD == 0:
            return 2 * jnp.ones(3)
        elif self.y0_iD == 1:
            return 20 * jnp.ones(3)
        elif self.y0_iD == 2:
            return 80 * jnp.ones(3)
        else:
            raise ValueError("y0_iD must be 0, 1 or 2")

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        con1 = x1**2 + x2**2 + x3**2 - 25.0
        con2 = 8.0 * x1 + 14.0 * x2 + 7.0 * x3 - 56.0
        return jnp.array([con1, con2]), None

    def bounds(self):
        # From BOUNDS section: all variables are free (FR)
        return None

    def expected_result(self):
        # The expected result given in the original source. (Not given in SIF file.)
        return jnp.array([3.5121, 0.21699, 3.5522])

    def expected_objective_value(self):
        # From OBJECT BOUND section of SIF file, LO SOLTN value
        return jnp.array(961.71517219)
