import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DENSCHNA(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem A.

    This is a 2-dimensional unconstrained optimization problem with
    nonlinear terms including exponentials.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: OUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # The objective function includes terms with x1^2 and exp(x2)
        term1 = x1**2 + x2**2
        term2 = jnp.exp(x1**2 + x2**2)
        term3 = x1**2 * jnp.exp(x2)
        term4 = x2**2 * jnp.exp(x1)

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.array([0.0, 0.0])

    def expected_objective_value(self):
        # At the origin, all terms evaluate to 0 except exp(0) = 1
        return jnp.array(1.0)


# TODO: human review required
class DENSCHNB(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem B.

    This is a 2-dimensional unconstrained optimization problem with
    a product term (x1 - 2.0) * x2.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # The objective is a sum of squares
        term1 = x1**2 + x2**2
        term2 = (x1 - 2.0) * x2

        return term1 + term2**2

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # Based on the problem formulation, the minimum is at:
        return jnp.array([2.0, 0.0])

    def expected_objective_value(self):
        # At x = [2.0, 0.0], we have term1 = 4 and term2 = 0
        return jnp.array(4.0)


# TODO: human review required
class DENSCHNC(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem C.

    This is a 2-dimensional unconstrained optimization problem with
    squares of variables and an exponential term.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # The objective includes squared variables and an exponential
        term1 = x1**2
        term2 = x2**2
        term3 = jnp.exp(x1 - 1.0)

        return term1 + term2 + term3**2

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([2.0, 3.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at x1 = 1.0 (where exp(x1-1) = 1) and x2 = 0.0
        return jnp.array([1.0, 0.0])

    def expected_objective_value(self):
        # At x = [1.0, 0.0], we have term1 = 1, term2 = 0, and term3^2 = 1
        return jnp.array(2.0)


# TODO: human review required
class DENSCHND(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem D.

    This is a 3-dimensional unconstrained optimization problem with
    polynomial terms up to the fourth power.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-3-0
    """

    n: int = 3  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # The objective includes various polynomial terms
        term1 = x1**2 + x2**2 + x3**2  # Second-order terms
        term2 = (x1 * x2) ** 2  # Fourth-order cross term
        term3 = (x1 * x3) ** 2  # Fourth-order cross term
        term4 = (x2 * x3) ** 2  # Fourth-order cross term

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([10.0, 10.0, 10.0])

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.array([0.0, 0.0, 0.0])

    def expected_objective_value(self):
        # At the origin, all terms evaluate to 0
        return jnp.array(0.0)


# TODO: human review required
class DENSCHNE(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem E.

    This is a 3-dimensional unconstrained optimization problem with
    squares of variables and an exponential term.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AN-3-0
    """

    n: int = 3  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # The objective includes squared variables and an exponential
        term1 = x1**2 + x2**2  # Sum of first two squares
        term2 = jnp.exp(x3)  # Exponential term
        term3 = (x1 * jnp.exp(x3)) ** 2  # Cross term with exponential

        return term1 + term2 + term3

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([2.0, 3.0, -8.0])

    def args(self):
        return None

    def expected_result(self):
        # The objective is minimized when x1 = 0, x2 = 0, and x3 is very negative
        # However, we can't reach -∞ in practice, so we'll use a large negative value
        return jnp.array([0.0, 0.0, -jnp.inf])

    def expected_objective_value(self):
        # As x3 approaches -∞, exp(x3) approaches 0, making the objective approach 0
        return jnp.array(0.0)


# TODO: human review required
class DENSCHNF(AbstractUnconstrainedMinimisation):
    """Dennis-Schnabel problem F.

    This is a 2-dimensional unconstrained optimization problem with
    a sum of squares formulation.

    Source: Problem from "Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations" by J.E. Dennis and R.B. Schnabel, 1983.

    Classification: SUR2-AY-2-0
    """

    n: int = 2  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # The objective is a sum of squares
        term1 = (x1 + x2) ** 2  # (x1 + x2)^2
        term2 = (x1 - x2) ** 2  # (x1 - x2)^2
        term3 = (x1 - 0.0) ** 2  # (x1 - 0)^2
        term4 = (x2 - 3.0) ** 2  # (x2 - 3)^2

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial values based on problem specification
        return jnp.array([2.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # Based on the objective formulation, the minimum occurs at:
        # - term1 and term2 want x1 and x2 to be 0 (which makes both terms 0)
        # - term3 wants x1 to be 0
        # - term4 wants x2 to be 3
        # This creates a trade-off between term2 and term4
        # The analytical solution can be computed by solving the system of equations
        return jnp.array([0.0, 1.5])

    def expected_objective_value(self):
        # At x = [0.0, 1.5], we have:
        # term1 = (0 + 1.5)^2 = 2.25
        # term2 = (0 - 1.5)^2 = 2.25
        # term3 = (0 - 0)^2 = 0
        # term4 = (1.5 - 3)^2 = (-1.5)^2 = 2.25
        return jnp.array(6.75)
