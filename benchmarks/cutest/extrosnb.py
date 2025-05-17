import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class EXTROSNB(AbstractUnconstrainedMinimisation):
    """Extended Rosenbrock function (nonseparable version).

    This is an extension of the classic Rosenbrock function to n dimensions.
    The function is characterized by a curved narrow valley, which makes it
    challenging for many optimization algorithms.

    The objective function is:
    f(x) = sum_{i=0}^{n-2} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Source: problem 21 in
    J.J. Moré, B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 100  # Default dimension (5, 10, 100, or 1000)

    def objective(self, y, args):
        del args

        # Define the function to compute the Rosenbrock term for a given index
        def rosenbrock_term(i):
            # 100(y_{i+1} - y_i^2)^2 + (1 - y_i)^2
            term1 = 100.0 * (y[i + 1] - y[i] ** 2) ** 2
            term2 = (1.0 - y[i]) ** 2

            return term1 + term2

        # Create an array of indices (0 to n-2)
        indices = jnp.arange(0, self.n - 1)

        # Compute all terms using vmap
        terms = jax.vmap(rosenbrock_term)(indices)

        # Sum all terms
        return jnp.sum(terms)

    def y0(self):
        # Starting point from the SIF file: all variables = -1.0
        return jnp.full(self.n, -1.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # The minimum objective value is 0.0
        return jnp.array(0.0)
