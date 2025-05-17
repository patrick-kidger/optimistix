import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HAIRY(AbstractUnconstrainedMinimisation, strict=True):
    """A hairy problem in two variables.

    The surface defined by this function has a large number of relatively
    sharp hills between which a valley leads to the minimizer.
    This problem contains a large number of saddle points.

    Dedicated to Meret Oppenheim, creator of the "furry cup" (1936).

    Source: Ph. Toint, private communication.
    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AY-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Constants from the SIF file (lines 24-25)
        hlength = 30.0
        cslope = 100.0

        # Parameters from element uses (lines 62, 67, 71)
        dens = 7.0
        smooth = 0.01

        # The fur component (lines 119-133)
        dv1 = dens * x1
        dv2 = dens * x2
        # Note: tdv1 and tdv2 are defined in the SIF file but not used in the objective
        # tdv1 = dv1 + dv1  # 2 * dens * x1
        # tdv2 = dv2 + dv2  # 2 * dens * x2
        s1sq = jnp.sin(dv1) ** 2
        c2sq = jnp.cos(dv2) ** 2
        fur_term = s1sq * c2sq

        # The 2D diagonal cup (lines 138-145)
        v_dcup = x1 - x2
        vsq_dcup = v_dcup**2
        arg_dcup = smooth + vsq_dcup
        dcup_term = jnp.sqrt(arg_dcup)

        # The 1D cup (lines 150-156)
        vsq_1cup = x1**2
        arg_1cup = smooth + vsq_1cup
        cup1d_term = jnp.sqrt(arg_1cup)

        # Group uses (lines 75-77)
        return hlength * fur_term + cslope * dcup_term + cslope * cup1d_term

    def y0(self):
        # Initial point from SIF file (lines 42-43)
        return jnp.array([-5.0, -7.0])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to SIF file (line 83), the optimal objective value is 20.0
        return jnp.array(20.0)


# TODO: human review required
class LOGHAIRY(AbstractUnconstrainedMinimisation, strict=True):
    """A more difficult variant of the HAIRY problem in two variables.

    It is defined by a logarithmic transformation of the HAIRY surface,
    which has a large number of relatively sharp hills between which a valley
    leads to the minimizer. This problem contains a large number of saddle points.

    The problem is nonconvex.

    Source: Ph. Toint, private communication.
    SIF input: Ph. Toint, April 1997.

    Classification: OUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Constants from the SIF file (lines 24-25)
        hlength = 30.0
        cslope = 100.0

        # Parameters from element uses (lines 62, 67, 71)
        dens = 7.0
        smooth = 0.01

        # The fur component (lines 119-133)
        dv1 = dens * x1
        dv2 = dens * x2
        # Note: tdv1 and tdv2 are defined in the SIF file but not used in the objective
        # tdv1 = dv1 + dv1  # 2 * dens * x1
        # tdv2 = dv2 + dv2  # 2 * dens * x2
        s1sq = jnp.sin(dv1) ** 2
        c2sq = jnp.cos(dv2) ** 2
        fur_term = s1sq * c2sq

        # The 2D diagonal cup (lines 138-145)
        v_dcup = x1 - x2
        vsq_dcup = v_dcup**2
        arg_dcup = smooth + vsq_dcup
        dcup_term = jnp.sqrt(arg_dcup)

        # The 1D cup (lines 150-156)
        vsq_1cup = x1**2
        arg_1cup = smooth + vsq_1cup
        cup1d_term = jnp.sqrt(arg_1cup)

        # Group uses (lines 75-77) - The HAIRY objective
        hairy_obj = hlength * fur_term + cslope * dcup_term + cslope * cup1d_term

        # Now apply the logarithmic transformation from lines 182-184
        s = 100.0  # From line 182
        return jnp.log((s + hairy_obj) / s)

    def y0(self):
        # Initial point from SIF file (lines 43-44)
        return jnp.array([-500.0, -700.0])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to SIF file (line 90),
        # the optimal objective value is approximately 0.1823216
        return jnp.array(0.1823216)
