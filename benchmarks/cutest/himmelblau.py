import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Needs verification against another CUTEst interface
class HIMMELBG(AbstractUnconstrainedMinimisation, strict=True):
    """The HIMMELBG function.

    A 2-variable problem by Himmelblau.

    Source: problem 33 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.
    See Buckley#87 (p. 67)

    SIF input: Ph. Toint, Dec 1989.
    Classification: OUR2-AN-2-0
    """

    def objective(self, y, args):
        del args
        x1, x2 = y
        exponential = jnp.exp(-x1 - x2)
        quadratic = 2 * x1**2 + 3 * x2**2
        return exponential * quadratic

    def y0(self):
        return jnp.array([0.5, 0.5])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)


# TODO: human review required
class HIMMELBB(AbstractUnconstrainedMinimisation, strict=True):
    """The HIMMELBB function.

    A 2-variable problem by Himmelblau.

    Source: problem 33 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    SIF input: Ph. Toint, Dec 1989.
    Classification: OUR2-AN-2-0
    """

    def objective(self, y, args):
        del args
        x1, x2 = y
        term1 = x1 * x2 * (1 - x1)
        term2 = 1 - x2 - (x1 * (1 - x1) ** 5)
        return term1 * term2

    def y0(self):
        return jnp.array([-1.2, 1.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: human review required
class HIMMELBCLS(AbstractUnconstrainedMinimisation, strict=True):
    """Himmelblau's nonlinear least-squares problem.

    A 2-variable problem with 2 residuals.

    Source: problem 201 in
    W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lecture Notes in Economics and Mathematical Systems 187,
    Springer Verlag, Berlin, 1981.

    SIF input: Ph. Toint, Dec 1989.
    Classification: SUR2-AN-2-0
    """

    def objective(self, y, args):
        del args
        x1, x2 = y

        # The residuals are the difference between x1^2 and 7, and x2^2 and 11
        r1 = x1**2 - 7
        r2 = x2**2 - 11

        # Return the sum of squares
        return r1**2 + r2**2

    def y0(self):
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The exact solution is (±√7, ±√11)
        # We'll return one of the four solutions
        return jnp.array([jnp.sqrt(7.0), jnp.sqrt(11.0)])

    def expected_objective_value(self):
        # At the optimal solution, both residuals are 0
        return jnp.array(0.0)


# TODO: human review required
class HIMMELBF(AbstractUnconstrainedMinimisation, strict=True):
    """Himmelblau's HIMMELBF function.

    A 4-variable least squares data fitting problem.

    Source: problem 33 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    SIF input: Ph. Toint, Dec 1989.
    Classification: SUR2-AN-4-0
    """

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Data points provided in the SIF file
        a_values = jnp.array(
            [0.0, 0.000428, 0.001000, 0.001610, 0.002090, 0.003480, 0.005250]
        )
        b_values = jnp.array([7.391, 11.18, 16.44, 16.20, 22.20, 24.02, 31.32])

        # Compute the residuals for each data point
        def residual_fn(a, b):
            numerator = x1**2 + a * x2**2 + a**2 * x3**2
            denominator = b * (1 + a * x4**2)
            return (numerator / denominator) - 1.0

        # Vectorize the computation across all data points
        residuals = jnp.array([residual_fn(a, b) for a, b in zip(a_values, b_values)])

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        return jnp.array([2.7, 90.0, 1500.0, 10.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: human review required
class HIMMELBH(AbstractUnconstrainedMinimisation, strict=True):
    """Himmelblau's HIMMELBH function.

    A 2-variable problem.

    Source: problem 33 in
    D.H. Himmelblau,
    "Applied nonlinear programming",
    McGraw-Hill, New-York, 1972.

    SIF input: Ph. Toint, Dec 1989.
    Classification: OUR2-AN-2-0
    """

    def objective(self, y, args):
        del args
        x1, x2 = y

        term1 = x1**3 - 3 * x1
        term2 = x2**2 - 2 * x2
        constant = 2

        return term1 + term2 + constant

    def y0(self):
        return jnp.array([0.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        # The solution is at x1 = 1, x2 = 1
        return jnp.array([1.0, 1.0])

    def expected_objective_value(self):
        # Evaluating at (1, 1): 1³ - 3*1 + 1² - 2*1 + 2 = 1 - 3 + 1 - 2 + 2 = -1
        return jnp.array(-1.0)
