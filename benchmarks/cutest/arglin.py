import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ARGLINA(AbstractUnconstrainedMinimisation, strict=True):
    """ARGLINA function.

    Variable dimension full rank linear problem.

    Source: Problem 32 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#80 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 200  # SIF file suggests 10, 50, 100, or 200
    m: int = 400  # SIF file suggests m >= n and values like 20, 100, 200, or 400

    def objective(self, y, args):
        del args
        n = self.n
        m = self.m

        # Compute the residuals for each equation
        residuals = jnp.zeros(m)

        # First n residuals (one per variable)
        for i in range(n):
            # For each equation i (i < n), calculate:
            # g_i = sum_j (-2/m) * x_j for j != i, and (1-2/m) * x_i for j == i
            g_i = 0.0
            for j in range(n):
                if j == i:
                    g_i += (1.0 - 2.0 / m) * y[j]
                else:
                    g_i += (-2.0 / m) * y[j]
            residuals = residuals.at[i].set(g_i)

        # Remaining m-n residuals
        for i in range(n, m):
            # For each equation i (i >= n), calculate:
            # g_i = sum_j (-2/m) * x_j for all j
            g_i = jnp.sum((-2.0 / m) * y)
            residuals = residuals.at[i].set(g_i)

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1.0 as specified in the SIF file
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: needs human review
class ARGLINB(AbstractUnconstrainedMinimisation, strict=True):
    """ARGLINB function.

    Variable dimension rank one linear problem.

    Source: Problem 33 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#93 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 200  # SIF file suggests 10, 50, 100, or 200
    m: int = 400  # SIF file suggests m >= n and values like 20, 100, 200, or 400

    def objective(self, y, args):
        del args
        n = self.n
        m = self.m

        # Compute the residuals for each equation
        residuals = jnp.zeros(m)

        # Each residual g_i is a weighted sum of all variables
        # g_i = sum_j (i*j) * x_j
        for i in range(m):
            g_i = 0.0
            for j in range(n):
                g_i += ((i + 1) * (j + 1)) * y[j]
            residuals = residuals.at[i].set(g_i)

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1.0 as specified in the SIF file
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # The SIF file comments mention:
        # *LO SOLTN(10)          4.6341D+00
        # *LO SOLTN(50)          24.6268657
        # *LO SOLTN(100)         49.6259352
        # But no value for n=200 is provided
        return None


# TODO: needs human review
class ARGLINC(AbstractUnconstrainedMinimisation, strict=True):
    """ARGLINC function.

    Variable dimension rank one linear problem, with zero rows and columns.

    Source: Problem 34 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#101 (with different N and M)
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 200  # SIF file suggests 10, 50, 100, or 200
    m: int = 400  # SIF file suggests m >= n and values like 20, 100, 200, or 400

    def objective(self, y, args):
        del args
        n = self.n
        m = self.m

        # Compute the residuals for each equation
        residuals = jnp.zeros(m)

        # First residual g_1 = 0
        # Last residual g_m = 0
        # For residuals g_i (1 < i < m), compute:
        # g_i = sum_j ((i-1)*(j)) * x_j for 1 < j < n
        for i in range(1, m - 1):
            g_i = 0.0
            for j in range(1, n - 1):
                g_i += ((i) * (j)) * y[j]
            residuals = residuals.at[i].set(g_i)

        # Sum of squares of residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial value of 1.0 as specified in the SIF file
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        # The SIF file comments mention:
        # *LO SOLTN(10)           6.13513513
        # *LO SOLTN(50)           26.1269035
        # *LO SOLTN(100)          26.1269
        # But no value for n=200 is provided
        return None
