import jax.numpy as jnp

from .problem import AbstractConstrainedMinimisation


class SNAKE(AbstractConstrainedMinimisation):
    """The SNAKE problem from the CUTEST collection of benchmark problems.

    This bivariate problem features a very nonconvex feasible region limited by two
    nearly parallel sine curves. The solution lies at the origin where these curves
    intersect. The angle at which they intersect (and thus the conditioning of the
    constraint's Jacobian) is controlled by the positive parameter TIP.

    The problem is not convex.

    Source: A problem designed by Ph. Toint for experimenting with feasibility issues in
    barrier approaches to nonlinear inequality constraints.

    Note J. Haffner --------------------------------------------------------------------
    This problem can show a substantial pessimisation. Using second-order corrections is
    essential to get a good solution in a small number of iterations in IPOPT.
    ------------------------------------------------------------------------------------

    SIF input: Ph.L. Toint, September 93.
    Classification: LOR2-AN-2-2
    """

    tip: float = 1e-4

    def objective(self, y, args):
        del args
        x1, _ = y
        return x1

    def y0(self):
        return jnp.array([1.0, 5.0])

    def args(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        g1 = x2 - jnp.sin(x1)  # x2 greater than sin(x1)
        g2 = jnp.sin(x1 * (1 + self.tip)) - x2  # x2 less than sin(x1 * (1 + TIP))
        return None, (g1, g2)

    def bounds(self):
        return None

    def expected_result(self):
        return jnp.array([0.0, 0.0])  # Solution at the origin

    def expected_objective_value(self):
        return jnp.array(0.0)  # Minimum objective value is 0.0
