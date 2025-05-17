import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ALLINITU(AbstractUnconstrainedMinimisation, strict=True):
    """The ALLINITU function.

    A problem with "all in it". Intended to verify that changes to LANCELOT are safe.

    Source: N. Gould, private communication.
    SIF input: Nick Gould, June 1990.

    Classification: OUR2-AY-4-0
    """

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Objective function is constructed based on the SIF file
        # Groups with their corresponding elements and mathematical operations

        # Default group type is L2 (GVAR * GVAR) unless specified as TRIVIAL (GVAR)

        # FT1 (TRIVIAL) - no element
        ft1 = 0.0

        # FT2 (TRIVIAL) - constant term (1.0) plus x3
        ft2 = 1.0 + x3

        # FT3 (TRIVIAL) - element FT3E1: SQR(x1)
        ft3 = x1**2

        # FT4 (TRIVIAL) - elements FT4E1: SQR(x2) and FT4E2: SQR2(x3 + x4)
        ft4 = x2**2 + (x3 + x4) ** 2

        # FT5 (TRIVIAL) - constant term (3.0) plus elements:
        # FT56E1: SINSQR(x3) and FT5E2: PRODSQR(x1, x2)
        ft5 = 3.0 + jnp.sin(x3) ** 2 + (x1**2 * x2**2)

        # FT6 (TRIVIAL) - element FT56E1: SINSQR(x3)
        ft6 = jnp.sin(x3) ** 2

        # FNT1 - no element, uses default group type L2 (GVAR * GVAR)
        fnt1 = 0.0**2

        # FNT2 - constant term (1.0) plus x4, uses default group type L2
        fnt2 = (1.0 + x4) ** 2

        # FNT3 - element FNT3E1: SQR(x2), uses default group type L2
        fnt3 = (x2**2) ** 2

        # FNT4 - elements FNT4E1: SQR(x3) and FNT4E2: SQR2(x4 + x1),
        # uses default group type L2
        fnt4 = (x3**2 + (x4 + x1) ** 2) ** 2

        # FNT5 - constant term (4.0) plus x1 plus elements:
        # FNT56E1: SINSQR(x4) and FNT5E2: PRODSQR(x2, x3)
        fnt5 = (4.0 + x1 + jnp.sin(x4) ** 2 + (x2**2 * x3**2)) ** 2

        # FNT6 - element FNT56E1: SINSQR(x4), uses default group type L2
        fnt6 = (jnp.sin(x4) ** 2) ** 2

        # Sum all group contributions
        return (
            ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + fnt1 + fnt2 + fnt3 + fnt4 + fnt5 + fnt6
        )

    def y0(self):
        # Initial point is not explicitly given in the SIF file
        # Using zeros as a reasonable starting point
        return jnp.zeros(4)

    def args(self):
        return None

    def expected_result(self):
        # No expected result is given in the SIF file
        return None

    def expected_objective_value(self):
        # No expected objective value is given in the SIF file
        return None
