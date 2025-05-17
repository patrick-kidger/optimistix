import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HELIX(AbstractUnconstrainedMinimisation, strict=True):
    """The helix problem.

    This test function involves the computation of the arctangent function
    (theta = arctan(y/x)) which should be in [0, 2π), and represents a path
    around a helix.

    Source: problem 7 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-3-0
    """

    def objective(self, y, args):
        del args

        # Extract variables
        x1, x2, x3 = y

        # Compute theta (arctangent) in a stable way
        theta = jnp.arctan2(x2, x1)

        # Ensure theta is in [0, 2π) range
        # In JAX we need to handle this carefully with modular arithmetic
        theta = jnp.where(theta < 0, theta + 2.0 * jnp.pi, theta)

        # Compute the residuals
        r1 = 10.0 * (theta - 2.0 * jnp.pi * x3)
        r2 = 10.0 * (jnp.sqrt(x1**2 + x2**2) - 1.0)
        r3 = x3

        # Return sum of squared residuals
        return r1**2 + r2**2 + r3**2

    def y0(self):
        # Starting point from the SIF file
        return jnp.array([-1.0, 0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # From the SIF file and the problem description,
        # the solution is (cos(0), sin(0), 0) = (1, 0, 0)
        return jnp.array([1.0, 0.0, 0.0])

    def expected_objective_value(self):
        # At the optimal solution, all residuals should be zero
        return jnp.array(0.0)
