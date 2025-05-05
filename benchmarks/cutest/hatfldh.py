import jax.numpy as jnp

from .problem import AbstractConstrainedMinimisation


# TODO: This problem might be useful to validate second-order corrections. Without them,
# we do seem to take a lot of unproductive steps before suddenly improving dramatically.
class HATFLDH(AbstractConstrainedMinimisation, strict=True):
    """The HATFLDH problem from the CUTEST collection of benchmark problems.

    A test problem from the OPTIMA user manual.
    This is a nonlinear objective with linear constraints.

    Source: "The OPTIMA user manual (issue No.8, p. 91)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Classification: QLR2-AN-4-7
    """

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y
        return -x1 * x3 - x2 * x4

    def y0(self):
        return jnp.array([1.0, 5.0, 5.0, 1.0])  # This starting point is infeasible

    def args(self):
        return None

    def constraint(self, y):
        """The sums of all pairwise combinations of the variables must fall within a
        specified range of values. The feasible region is four-dimensional diamond,
        cut in half by the hyperplane specified through the sum constraint
        x1 + x2 + x3 + x4 >= 5.0.
        """
        A = jnp.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        pairwise_sums = jnp.matmul(A, y)

        constants = jnp.array([2.5, 2.5, 2.5, 2.0, 2.0, 1.5])
        ranges = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        lower_bounds = constants - 0.5 * ranges
        upper_bounds = constants + 0.5 * ranges

        distance_from_lower = pairwise_sums - lower_bounds
        distance_from_upper = upper_bounds - pairwise_sums

        sum_above = jnp.sum(y) - 5.0  # Verified against CUTEST.jl (J. Haffner)

        inequalities = (distance_from_lower, distance_from_upper, sum_above)
        return None, inequalities

    def bounds(self):
        lower = jnp.zeros(4)  # Verified against CUTEST.jl (J. Haffner)
        upper = jnp.full(4, 5.0)
        return lower, upper

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(-24.4999998)
