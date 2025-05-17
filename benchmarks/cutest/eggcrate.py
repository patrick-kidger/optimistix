import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


class EGGCRATE(AbstractUnconstrainedMinimisation, strict=True):
    """EGGCRATE function.

    A simple unconstrained nonlinear least squares problem with a periodic
    objective function that resembles an egg carton or egg crate in 3D.

    Source: problem 35 in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-MN-4-0
    """

    def objective(self, y, args):
        del args

        # Extract the two variables
        x, y = y

        # The objective function consists of 4 residuals:
        # f1 = x
        # f2 = y
        # f3 = 5*sin(x)
        # f4 = 5*sin(y)

        # Computing the sum of squares directly
        return x**2 + y**2 + 25.0 * jnp.sin(x) ** 2 + 25.0 * jnp.sin(y) ** 2

    def y0(self):
        # Starting point from the SIF file: (1.0, 2.0)
        return jnp.array([1.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is (0, 0)
        return jnp.zeros(2)

    def expected_objective_value(self):
        # The minimum objective value is 0.0
        return jnp.array(0.0)
