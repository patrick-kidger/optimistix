import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class GENHUMPS(AbstractUnconstrainedMinimisation):
    """GENHUMPS - Generalized HUMPS function.

    A multi-dimensional variant of HUMPS, a two dimensional function
    with a lot of humps. The density of humps increases with the
    parameter ZETA, making the problem more difficult.

    The problem is nonconvex.

    Source: Ph. Toint, private communication, 1997.
    SDIF input: N. Gould and Ph. Toint, November 1997.

    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # Default dimension (5, 10, 100, 500, 1000, 5000)
    zeta: float = 20.0  # Density of humps parameter

    def objective(self, y, args):
        del args

        # Define the function to compute the terms for each pair of adjacent variables
        def pair_term(i):
            # Extract the two variables for this term
            x_i = y[i]
            x_i_plus_1 = y[i + 1]

            # The main hump term: (sin(zeta*x_i) * sin(zeta*x_{i+1}))^2
            sine_term = (
                jnp.sin(self.zeta * x_i) * jnp.sin(self.zeta * x_i_plus_1)
            ) ** 2

            # The quadratic terms: 0.05 * (x_i^2 + x_{i+1}^2)
            quadratic_term = 0.05 * (x_i**2 + x_i_plus_1**2)

            return sine_term + quadratic_term

        # Compute all the terms using vmap
        indices = jnp.arange(0, self.n - 1)
        terms = jax.vmap(pair_term)(indices)

        # Sum all terms
        return jnp.sum(terms)

    def y0(self):
        # Starting point from the SIF file:
        # all variables = -506.2, except first = -506.0
        y_init = jnp.full(self.n, -506.2)
        y_init = y_init.at[0].set(-506.0)
        return y_init

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not specified in the SIF file
        return None

    def expected_objective_value(self):
        # The minimum objective value is given as 0.0
        return jnp.array(0.0)
