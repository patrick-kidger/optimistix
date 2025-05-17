import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HUMPS(AbstractUnconstrainedMinimisation, strict=True):
    """The HUMPS function.

    A two-dimensional function with many humps.
    The density of humps increases with parameter zeta,
    making the problem more difficult for optimization algorithms.

    Source:
    Ph. Toint,
    Private communication.

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-2-0
    """

    zeta: float = 20.0  # Parameter that controls density of humps

    def objective(self, y, args):
        del args
        x, y_val = y
        zeta = self.zeta

        # Sinusoidal term (humps)
        sin_term = jnp.sin(zeta * x) * jnp.sin(zeta * y_val)
        hump_term = sin_term**2

        # Quadratic term
        quad_term = 0.05 * (x**2 + y_val**2)

        # Full objective function
        return hump_term + quad_term

    def y0(self):
        # Starting point from SIF file
        return jnp.array([-506.0, -506.2])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is at the origin (0, 0)
        # since both terms are minimized there
        return jnp.array([0.0, 0.0])

    def expected_objective_value(self):
        # At (0, 0), the sin terms are 0, and the quadratic term is also 0
        return jnp.array(0.0)
