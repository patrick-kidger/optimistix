import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HATFLDD(AbstractUnconstrainedMinimisation, strict=True):
    """An exponential fitting test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 35)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.

    Classification: SUR2-AN-3-0
    """

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Time points
        t_values = jnp.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])

        # Data values
        z_values = jnp.array(
            [1.751, 1.561, 1.391, 1.239, 1.103, 0.981, 0.925, 0.8721, 0.8221, 0.7748]
        )

        # Compute model values for each data point
        # The model is x1 * exp(t * x2) - exp(t * x3)
        model_values = x1 * jnp.exp(t_values * x2) - jnp.exp(t_values * x3)

        # Compute residuals
        residuals = model_values - z_values

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (lines 75-77)
        return jnp.array([1.0, -1.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to SIF file (line 120), the optimal objective value is 6.615114D-08
        return jnp.array(6.615114e-08)


# TODO: human review required
class HATFLDE(AbstractUnconstrainedMinimisation, strict=True):
    """An exponential fitting test problem from the OPTIMA user manual.

    Source:
    "The OPTIMA user manual (issue No.8, p. 37)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.

    Classification: SUR2-AN-3-0
    """

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Time points
        t_values = jnp.array(
            [
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
                1.05,
                1.1,
                1.15,
                1.2,
                1.25,
                1.3,
            ]
        )

        # Data values
        z_values = jnp.array(
            [
                1.561,
                1.473,
                1.391,
                1.313,
                1.239,
                1.169,
                1.103,
                1.04,
                0.981,
                0.925,
                0.8721,
                0.8221,
                0.7748,
                0.73,
                0.6877,
                0.6477,
                0.6099,
                0.5741,
                0.5403,
                0.5084,
                0.4782,
            ]
        )

        # Compute model values for each data point
        # The model is x1 * exp(t * x2) - exp(t * x3)
        model_values = x1 * jnp.exp(t_values * x2) - jnp.exp(t_values * x3)

        # Compute residuals
        residuals = model_values - z_values

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (lines 97-99)
        return jnp.array([1.0, -1.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to SIF file (line 142), the optimal objective value is 5.120377D-07
        return jnp.array(5.120377e-07)


# TODO: human review required
class HATFLDFL(AbstractUnconstrainedMinimisation, strict=True):
    """Fletcher's variation of a test problem (HATFLDF) from the OPTIMA user manual.

    Monotonic paths to the solution from the initial point move to infinity and back.

    Source:
    "The OPTIMA user manual (issue No.8, p. 47)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990, mods Nick Gould, August 2008

    Nonlinear least-squares variant

    Classification: SUR2-AN-3-0
    """

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Constants from SIF file (lines 43-45)
        targets = jnp.array([0.032, 0.056, 0.099])

        # Parameters for the element uses
        t_values = jnp.array([1.0, 2.0, 3.0])

        # Compute model values for each data point
        # The model uses x1 + x2 * x3^t for each t in t_values
        # Note: XPEXP in this case is defined differently from HATFLDD and HATFLDE
        # Here, it's specifically x * y^t where t is an integer, not an exponential
        model_values = x1 + x2 * jnp.power(x3, t_values)

        # Compute residuals
        residuals = model_values - targets

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (the "nasty" starting point, lines 55-57)
        return jnp.array([1.2, -1.2, 0.98])

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Not provided in the SIF file
        return None


# TODO: human review required
class HATFLDFLS(AbstractUnconstrainedMinimisation, strict=True):
    """A test problem from the OPTIMA user manual.

    Least-squares version of HATFLDF.

    Source:
    "The OPTIMA user manual (issue No.8, p. 47)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Least-squares version of HATFLDF.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-AN-3-0
    """

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Constants from SIF file (lines 40-42)
        targets = jnp.array([0.032, 0.056, 0.099])

        # Parameters for the element uses
        t_values = jnp.array([1.0, 2.0, 3.0])

        # Compute model values for each data point
        # The model is x1 + x2 * exp(t * x3)
        model_values = x1 + x2 * jnp.exp(t_values * x3)

        # Compute residuals
        residuals = model_values - targets

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (line 50)
        return jnp.full(3, 0.1)

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Not provided in the SIF file
        return None


# TODO: human review required
class HATFLDGLS(AbstractUnconstrainedMinimisation, strict=True):
    """A test problem from the OPTIMA user manual.

    Least-squares version of HATFLDG.

    Source:
    "The OPTIMA user manual (issue No.8, p. 49)",
    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.

    SIF input: Ph. Toint, May 1990.
    Least-squares version of HATFLDG.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-AY-25-0
    """

    def objective(self, y, args):
        del args

        # Create residuals for each group
        # Each group is defined by G(i) = x(i) + x(13) - 1.0 for i=1...n
        residuals = y + y[12] - 1.0

        # Group element G(1) also involves A(1) with a -1.0 coefficient
        # A(1) is a 2PR element with x1 and x2, computing x1 * x2
        residuals = residuals.at[0].add(-1.0 * y[0] * y[1])

        # Groups G(2) to G(n) involve A(i) elements
        # For i=2 to n-1, A(i) is a 2PRI element with x(i), x(i-1), and x(i+1)
        # 2PRI computes (x(i) + 1.0) * (x(i-1) + 1.0 - x(i+1))

        # Use vmap instead of for-loop
        def compute_2pri_element(i):
            return (y[i] + 1.0) * (y[i - 1] + 1.0 - y[i + 1])

        indices = jnp.arange(1, len(y) - 1)
        inner_residuals = jax.vmap(compute_2pri_element)(indices)
        residuals = residuals.at[1:-1].add(inner_residuals)

        # For i=n, A(n) is a 2PR element with x(n-1) and x(n), computing x(n-1) * x(n)
        residuals = residuals.at[-1].add(y[-2] * y[-1])

        # Objective function is sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial point from SIF file (line 57)
        return jnp.ones(25)  # Hard-coded as 25 per SIF file

    def args(self):
        return None

    def expected_result(self):
        # Not provided in the SIF file
        return None

    def expected_objective_value(self):
        # Not provided in the SIF file
        return None
