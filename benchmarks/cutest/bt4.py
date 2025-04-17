import equinox as eqx
import jax.numpy as jnp


# TODO: with optx.IPOPTLike I currently get the correct result for the first and second
# starting points, but a runtime error for the third.
# TODO: problems should have an iD attribute that indicates a variant (e.g. starting
# point, dimensionality.) Look at how this is done in the Julia interface.
class BT4(eqx.Module):
    """The BT4 problem from the CUTEST collection of benchmark problems, as documented
    at https://bitbucket.org/optrove/sif. This particular problem is a variant of
    problem 4, originally described in https://www.jstor.org/stable/2157674?seq=21, on
    page 620.

    The original problem seems to be unbounded. The contribution of x3 in the first
    constraint has been squared instead of cubed. (Modification done in SIF collection.)
    The problem is not convex.

    SIF input: Ph. Toint, June 1993.
    Classification: QQR2-AN-3-2
    """

    y0_iD: int = 0  # Can also be 1, or 2

    def objective(self, y, args):
        del args
        x1, x2, _ = y
        return x1 - x2 + x2**3

    def y0(self):
        if self.y0_iD == 0:
            return jnp.array([3.1494, 1.4523, -3.6017])
        elif self.y0_iD == 1:
            return jnp.array([3.122, 1.489, -3.611])
        elif self.y0_iD == 2:
            return jnp.array([-0.94562, -2.35984, 4.30546])
        else:
            raise ValueError("Invalid start point ID. Valid iDs are 0, 1, or 2.")

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        con1 = x1**2 + x2**2 + x3**2 - 25.0
        con2 = x1 + x2 + x3 - 1.0
        return jnp.array([con1, con2]), None

    def bounds(self):
        # From BOUNDS section: all variables are free (FR)
        return None

    def expected_result(self):
        # TODO this is the result I get for the first and second starting point using
        # optx.IPOPTLike, the objective function value then matches what is given in the
        # SIF file. This should be verified against another interface to CUTEST.
        return jnp.array([3.6313505, 0.72778094, -3.3591316])

    def expected_objective_value(self):
        return jnp.array([3.28903771])
