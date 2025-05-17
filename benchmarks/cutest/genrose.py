import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class GENROSE(AbstractUnconstrainedMinimisation):
    """GENROSE - Generalized Rosenbrock function.

    This is a generalization of the Rosenbrock function to n dimensions.
    The objective function has the form:
    f(x) = 1 + sum_{i=2}^n (0.01 * (x_i - 1)^2 + 100 * (x_i - x_{i-1}^2)^2)

    Source: problem 5 in
    S. Nash,
    "Newton-type minimization via the Lanczos process",
    SIAM J. Num. Anal. 21, 1984, 770-788.

    SIF input: Nick Gould, Oct 1992.
              minor correction by Ph. Shott, Jan 1995.

    Classification: SUR2-AN-V-0
    """

    n: int = 500  # Default dimension (5, 10, 100, 500)

    def objective(self, y, args):
        del args

        # The generalized Rosenbrock function
        # f(x) = 1 + sum_{i=2}^n (0.01 * (x_i - 1)^2 + 100 * (x_i - x_{i-1}^2)^2)

        # First term: constant 1.0
        result = 1.0

        # Compute the quadratic terms (x_i - 1)^2 for i=2,...,n
        quadratic_terms = 0.01 * jnp.sum((y[1:] - 1.0) ** 2)

        # Compute the main Rosenbrock terms: (x_i - x_{i-1}^2)^2 for i=2,...,n
        def rosenbrock_term(i):
            return (y[i] - y[i - 1] ** 2) ** 2

        indices = jnp.arange(1, self.n)
        rosenbrock_terms = jax.vmap(rosenbrock_term)(indices)
        rosenbrock_sum = 100.0 * jnp.sum(rosenbrock_terms)

        # Sum all terms
        return result + quadratic_terms + rosenbrock_sum

    def y0(self):
        # Starting point from the SIF file: x_i = i/(n+1)
        indices = jnp.arange(1, self.n + 1)
        return indices / (self.n + 1)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is all variables = 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # The minimum objective value is given as 1.0
        return jnp.array(1.0)
