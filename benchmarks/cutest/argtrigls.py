import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ARGTRIGLS(AbstractUnconstrainedMinimisation, strict=True):
    """ARGTRIGLS function.

    Variable dimension trigonometric problem in least-squares form.
    This problem is a sum of n least-squares groups, each of
    which has n+1 nonlinear elements. Its Hessian matrix is dense.

    Source: Problem 26 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Least-squares version: Nick Gould, Oct 2015.

    Classification: SUR2-AN-V-0
    """

    n: int = 200  # SIF file suggests 10, 50, 100, or 200

    def objective(self, y, args):
        del args
        n = self.n

        # Compute the residuals for each equation
        residuals = jnp.zeros(n)

        for i in range(n):
            # Each residual g_i combines:
            # 1. n terms of cosine elements
            # 2. sincos element scaled by i

            # Compute sum of cosines for all variables
            sum_cos = jnp.sum(jnp.cos(y))

            # Compute sincos term for variable i
            sincos_i = (jnp.cos(y[i]) + jnp.sin(y[i])) * (i + 1)

            # Total residual is sincos_i + sum_cos
            # Note: constant term n+i is handled in the SIF file
            # but doesn't affect optimization
            residuals = residuals.at[i].set(sincos_i + sum_cos)

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1/n as specified in the SIF file
        return jnp.ones(self.n) / self.n

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # The SIF file comments mention: *LO SOLTN 0.0
        return jnp.array(0.0)
