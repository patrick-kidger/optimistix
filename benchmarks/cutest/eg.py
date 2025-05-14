import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


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
