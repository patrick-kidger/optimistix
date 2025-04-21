import equinox as eqx
import jax.numpy as jnp


class BT1(eqx.Module):
    """The BT1 problem from the CUTEST collection of benchmark problems.
    Original source: https://www.jstor.org/stable/2157674?seq=21, page 620.

    Some characteristics of this problem (Note added by Johanna Haffner):

    This problem consists of finding the minimum along a circular line over a
    paraboloid, where the circle is just slightly shifted so that it is almost tangent
    to the contour lines of the paraboloid. This creates an optimisation landscape that
    is relatively flat near the solution, which can lead to very slow convergence.
    Additionally, this problem is poorly scaled: the objective function multiplies the
    values of x1, x2 by a factor of 10, while the constraint function does not.
    The initial value is infeasible, and poor initialisation of the dual variables can
    lead to large initial steps in the dual variables, which can cause the Hessian of
    the Lagrangian to become negative-definite.

    # TODO add SIF entry
    # TODO add classifier
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
