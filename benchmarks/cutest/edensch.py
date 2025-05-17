import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


class EDENSCH(AbstractUnconstrainedMinimisation, strict=True):
    """Extended Dennis and Schnabel problem.

    This problem involves a sum of quartic terms for each variable
    and quadratic terms for products of adjacent variables.

    Source: problem 157 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    n: int = 2000  # Default dimension

    def objective(self, y, args):
        del args

        # The objective function for EDENSCH consists of two sums:
        # 1. Sum of (x_i + 2)^4 terms for all variables
        # 2. Sum of 4(x_i * x_{i+1})^2 terms for adjacent pairs

        # Compute the quartic terms: sum((x_i + 2)^4)
        quartic_terms = (y + 2.0) ** 4
        quartic_sum = jnp.sum(quartic_terms)

        # Compute the quadratic interaction terms: sum(4(x_i * x_{i+1})^2)
        # For variables 0 to n-2, multiply each with the next variable
        products = y[:-1] * y[1:]
        quadratic_terms = 4.0 * products**2
        quadratic_sum = jnp.sum(quadratic_terms)

        return quartic_sum + quadratic_sum

    def y0(self):
        # Starting point from the SIF file: all variables = 8.0
        return jnp.full(self.n, 8.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not specified in the SIF file
        return None

    def expected_objective_value(self):
        # Expected minimum value depends on dimension
        if self.n == 36:
            return jnp.array(219.28)
        elif self.n == 2000:
            return jnp.array(1.20032e4)
        return None
