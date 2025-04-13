import equinox as eqx
import jax.numpy as jnp


class BT1(eqx.Module):
    """The BT1 problem from the CUTEST collection of benchmark problems.
    Original source: https://www.jstor.org/stable/2157674?seq=21, page 620.
    This problem consists of finding the minimum along a circular line over a
    paraboloid, where the circle is just slightly shifted so that it is almost tangent
    to the contour lines of the paraboloid. This creates a Lagrangian with a large and
    relatively flat region, where the worst case behaviour consists of taking many tiny
    steps and thus converging very slowly.
    """

    def objective(self, y, args):
        x1, x2 = y
        return -x1 + 10 * (x1**2 + x2**2 - 1)

    def y0(self):
        return jnp.array([0.08, 0.06])

    def args(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        return x1**2 + x2**2 - 1, None

    def bounds(self):
        return None

    def expected_result(self):
        return jnp.array([1.0, 0.0])
