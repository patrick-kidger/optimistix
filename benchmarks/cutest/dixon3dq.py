import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DIXON3DQ(AbstractUnconstrainedMinimisation):
    """Dixon's tridiagonal quadratic optimization problem.

    This is a variable-dimension quadratic problem that involves a
    tridiagonal structure in the objective function. The objective is of the form:
    (x_1)^2 + sum_{i=2}^{n-1} (x_i - x_{i+1})^2 + (x_n)^2

    Source: problem 156 (p. 51) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    n: int = 10000  # Default dimension

    def __init__(self, n=None):
        if n is not None:
            self.n = n

    def objective(self, y, args):
        del args

        # First term: x_1^2
        term1 = y[0] ** 2

        # Middle terms: sum_{i=2}^{n-1} (x_i - x_{i+1})^2
        indices1 = jnp.arange(1, self.n - 1)
        indices2 = indices1 + 1
        term2 = jnp.sum((y[indices1] - y[indices2]) ** 2)

        # Last term: x_n^2
        term3 = y[-1] ** 2

        return term1 + term2 + term3

    def y0(self):
        # Initial values from SIF file: all -1.0
        return jnp.full(self.n, -1.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)
