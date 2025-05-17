import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


class DQDRTIC(AbstractUnconstrainedMinimisation, strict=True):
    """A simple diagonal quadratic optimization test problem.

    This problem is a sum of N-2 squared terms, where each term involves three variables
    with specific coefficients.

    Source: problem 22 in
    Ph. L. Toint,
    "Test problems for partially separable optimization and results
    for the routine PSPMIN",
    Report 83/4, Department of Mathematics, FUNDP (Namur, B), 1983.

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    n: int = 5000  # Default dimension (other options: 10, 50, 100, 500, 1000)

    def objective(self, y, args):
        del args

        # The objective function for DQDRTIC is a sum of squared terms,
        # where each term is [X(i) - 0.01*X(i+1) - 0.01*X(i+2)]²

        # We'll compute the terms for indices 0 to n-3
        def term(i):
            return (y[i] - 0.01 * y[i + 1] - 0.01 * y[i + 2]) ** 2

        indices = jnp.arange(self.n - 2)
        terms = jax.vmap(term)(indices)

        return jnp.sum(terms)

    def y0(self):
        # Starting point from the SIF file: all variables = 3.0
        return jnp.full(self.n, 3.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is all variables = 0
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # The minimum objective value is 0.0
        return jnp.array(0.0)


class DQRTIC(AbstractUnconstrainedMinimisation, strict=True):
    """DQRTIC function - A diagonal quartic optimization test problem.

    This problem is a sum of quartic terms, where each term is of the form (x_i - i)^4.

    Source: problem 157 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    n: int = 5000  # Default dimension (other options: 10, 50, 100, 500, 1000)

    def objective(self, y, args):
        del args

        # The objective function for DQRTIC is a sum of quartic terms,
        # where each term is (x_i - i)^4
        # Note: SIF uses 1-based indexing, so we adjust by adding 1 to our indices
        indices = jnp.arange(1, self.n + 1)

        # Calculate (x_i - i)^4 for each i
        terms = (y - indices) ** 4

        return jnp.sum(terms)

    def y0(self):
        # Starting point from the SIF file: all variables = 2.0
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is x_i = i for all i
        return jnp.arange(1, self.n + 1)

    def expected_objective_value(self):
        # The minimum objective value is 0.0
        return jnp.array(0.0)
