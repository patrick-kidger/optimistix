import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class EG1(AbstractUnconstrainedMinimisation, strict=True):
    """The EG1 function.

    A simple nonlinear problem given as an example in Section 1.2.3 of
    the LANCELOT Manual.

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "LANCELOT, A Fortran Package for Large-Scale Nonlinear Optimization
    (Release A)" Springer Verlag, 1992.

    SIF input: N. Gould and Ph. Toint, June 1994.

    Classification: OBR2-AY-3-0
    """

    n: int = 3  # Problem has 3 variables

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # GROUP1: x1^2
        f1 = x1**2

        # GROUP2: (x2*x3)^4
        f2 = (x2 * x3) ** 4

        # GROUP3: combination of sin(x2 + x1 + x3) and x1*x3
        f3_1 = x2 * jnp.sin(x2 + x1 + x3)
        f3_2 = x1 * x3

        return f1 + f2 + f3_1 + f3_2

    def y0(self):
        # Initial values from SIF file
        return jnp.array([1.0, 1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class EG2(AbstractUnconstrainedMinimisation, strict=True):
    """The EG2 function.

    A simple nonlinear problem given as an example in Section 1.2.4 of
    the LANCELOT Manual. The problem is non convex and has several local minima.

    Source:
    A.R. Conn, N. Gould and Ph.L. Toint,
    "LANCELOT, A Fortran Package for Large-Scale Nonlinear Optimization
    (Release A)" Springer Verlag, 1992.

    Note J. Haffner --------------------------------------------------------------------
    Reference: https://doi.org/10.1007/978-3-662-12211-2_1, Chapter 1, page 11
    ------------------------------------------------------------------------------------

    SIF input: N. Gould and Ph. Toint, June 1994.

    Classification: OUR2-AN-1000-0
    """

    n: int = 1000  # Problem specifies N=1000

    def objective(self, y, args):
        # Verified against original reference (J. Haffner)
        del args
        first = y[0]
        last = y[-1]
        f1 = jnp.sum(jnp.sin(last**2 + y[:-1] ** 2 + first - 1))
        f2 = 0.5 * jnp.sin(last**2)
        return f1 + f2

    def y0(self):
        # Initial guess - not specified in the problem,
        # using all ones as a reasonable starting point
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
