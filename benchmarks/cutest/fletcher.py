import equinox as eqx
import jax.numpy as jnp


# TODO: I can't solve this problem yet with the IPOPTLike solver.
class FLETCHER(eqx.Module):
    """The FLETCHER problem from the CUTEST collection of benchmark problems.

    A problem found in Fletcher's book. The problem is to find the least area of a
    right-angled triangle which contains a circle of unit radius.

    Note J. Haffner: -------------------------------------------------------------------
    In this problem, we use two points to describe the vertices of a triangle. Assuming
    that the triangle is in the first quadrant, we can draw the problem as follows:

    (0, x2) ________________ (x1, x2)
            |  .      * *  |
            |    .  *  p  *|    with p = (x3, x4)
            |       . * *  |
            |          .   |
            |            . |
            ---------------- (x1, 0)

    We define an equality constraint in terms of the distance of the center of the point
    to the hypotenus of the triangle, setting it to 1.0 - this defines the unit radius.
    To make sure that the circle is inside the triangle, we define three inequality
    constraints (in SIF file, the book lists four, but one appears redundant). Two of
    these constrain the position of the center of the circle with respect to the
    vertices, while the third one restricts the movement of the center of the circle
    to the half plane in which x3 > x4. A fourth constraint is specified as a bound on
    x4, restricting to to values >= 1.

    The starting point is infeasible with respect to all constraints, except for the
    active bound constraint on x4 and the inequality constraint x3 >= x4, which is
    active. For interior point methods, this starting point may require a modification,
    since the barrier terms diverge as the (constraint) bounds are approached.
    ------------------------------------------------------------------------------------

    Source: R. Fletcher, "Practical Methods of Optimization", 2nd edition, Wiley, 1987.
    Available here: https://doi.org/10.1002/9781118723203.ch12, page 330.

    SIF input: Ph. Toint, March 1994.
    Classification: QOR2-AN-4-4
    """

    def objective(self, y, args):
        del args
        x1, x2, *_ = y
        return x1 * x2  # Factor of 1/2 (for triangle) can be dropped

    def y0(self):
        return jnp.ones(4)

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3, x4 = y

        # Equality constraint: circle must touch the hypotenuse of the triangle
        c1 = (x1 * x3 + x2 * x4) ** 2 / (x1**2 + x2**2) - x3**2 - x4**2 + 1.0

        # Inequality constraints:
        c2 = x1 - x3 - 1.0  # x1 >= x3 + 1
        c3 = x2 - x4 - 1.0  # x2 >= x4 + 1
        c4 = x3 - x4  # x3 >= x4
        c5 = x4 - 1.0  # x4 >= 1.0  # TODO: currently XDYcYd does not handle bounds

        return jnp.array([c1]), jnp.array([c2, c3, c4, c5])

    def bounds(self):
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, 1.0])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        return lower, upper

    def expected_result(self):
        # The optimal point coordinates are not provided in the file
        return None

    def expected_objective_value(self):
        # Not provided in the file
        return None
