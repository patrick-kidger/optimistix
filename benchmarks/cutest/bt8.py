import equinox as eqx
import jax.numpy as jnp


class BT8(eqx.Module):
    """The BT8 problem from the CUTEST collection of benchmark problems. Parsed from the
    file given in https://bitbucket.org/optrove/sif, original source given in
    https://www.jstor.org/stable/2157674?seq=21, on page 621.

    The first constraint has been edited to conform to what is given in the original
    source, I think the SIF file lists it as using the square of x2 instead of x1 (?).
    (This is my interpretation of the GROUP USES and ELEMENT USES sections, which might
    be wrong.)

    The problem is not convex.

    SIF input: Ph. Toint, June 1993.
    Classification: QQR2-AN-5-2
    """

    y0_iD: int = 0

    def __check_init__(self):
        assert self.y0_iD in (0, 1), "y0_iD must be 0 or 1."

    def objective(self, y, args):
        del args
        x1, x2, x3, _, _ = y
        return x1**2 + x2**2 + x3**2

    def y0(self):
        if self.y0_iD == 0:
            return jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
        elif self.y0_iD == 1:
            return 7 * jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
        else:
            raise ValueError("y0_iD must be 0 or 1.")

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, _, x4, x5 = y
        con1 = x1 - x4**2 - 1.0  # As given in original source
        con2 = x1**2 + x2**2 - x5**2 - 1.0
        return jnp.array([con1, con2]), None

    def bounds(self):
        # From BOUNDS section: all variables are free (FR)
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])  # as given in original source

    def expected_objective_value(self):
        return jnp.array(1.0)
