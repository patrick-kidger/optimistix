import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


class DQRTIC(AbstractUnconstrainedMinimisation, strict=True):
    """The DQRTIC function.

    Variable dimension diagonal quartic problem.

    Source: problem 157 (p. 87) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # Default dimension from SIF file is 5000, but can be changed

    def objective(self, y, args):
        del args
        # From the SIF file, the objective is sum[(x_i - i)^4] for i=1...n
        i_values = jnp.arange(1, self.n + 1)
        return jnp.sum((y - i_values) ** 4)

    def y0(self):
        # Initial values from SIF file: all variables = 2.0
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is x_i = i for all i
        return jnp.arange(1, self.n + 1)

    def expected_objective_value(self):
        # When x_i = i, the objective value is 0
        return jnp.array(0.0)
