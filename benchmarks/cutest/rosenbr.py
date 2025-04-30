import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


class ROSENBR(AbstractUnconstrainedMinimisation, strict=True):
    """The Rosenbrock "banana valley" problem from the CUTEST collection of benchmark
    problems.

    Original source: problem 1 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Classification: SUR2-AN-2-0
    """

    def objective(self, y, args):
        x1, x2 = y
        return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2

    def y0(self):
        return jnp.array([-1.2, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 1.0])

    def expected_objective_value(self):
        return jnp.array(0.0)
