import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOX(AbstractUnconstrainedMinimisation, strict=True):
    """A quartic function with a non-trivial sparsity pattern.

    Source: N. Gould, private communication.
    SIF input: N. Gould, Jan 2009

    Classification: OUR2-AN-V-0
    """

    def __init__(self, n=10000):
        """Initialize the BOX problem.

        Args:
            n: Dimension of the problem. Original value is 10000,
               other values are 10, 100, 1000, and 100000.
        """
        self.n = n

    def objective(self, y, args):
        del args
        n = self.n
        n_half = n // 2

        # Sum of (x_i + x_1)^2 terms
        a_terms = jnp.sum((y + y[0]) ** 2)

        # Sum of (x_i + x_n)^2 terms
        b_terms = jnp.sum((y + y[-1]) ** 2)

        # Sum of (x_i + x_{n/2})^2 terms
        c_terms = jnp.sum((y + y[n_half - 1]) ** 2)

        # Sum of (-0.5*x_i)^2 terms
        d_term = jnp.sum((-0.5 * y) ** 2)

        # Sum of x_i^4 terms
        q_terms = jnp.sum(y**4)

        return a_terms + b_terms + c_terms + d_term + q_terms

    def y0(self):
        # Initial values not specified in SIF file
        # Using default all zeros as a reasonable starting point
        return jnp.zeros(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The optimal objective value bound is -(n-1) according to line 74
        return jnp.array(-(self.n - 1))


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOX3(AbstractUnconstrainedMinimisation, strict=True):
    """Box problem in 3 variables.

    This function is a nonlinear least squares with 10 groups. Each
    group has 2 nonlinear elements of exponential type.

    Source: Problem 12 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#BOX663
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-3-0
    """

    n: int = 3  # Problem has 3 variables
    m: int = 10  # Number of data points

    def objective(self, y, args):
        del args
        x1, x2, x3 = y

        # Define indices from 1 to m
        indices = jnp.arange(1, self.m + 1)

        # Define inner function to compute residual for a single index i
        def compute_residual(i):
            t_i = -0.1 * i

            # Compute exp(-0.1*i*x1) - exp(-0.1*i*x2)
            term1 = jnp.exp(t_i * x1) - jnp.exp(t_i * x2)

            # Compute coefficient: -exp(-0.1*i) + exp(-i)
            coeff = -jnp.exp(t_i) + jnp.exp(-i)

            # Compute the residual: x3 * coeff + term1
            residual = x3 * coeff + term1

            return residual**2

        # Vectorize the function over indices and sum the results
        residuals = jax.vmap(compute_residual)(indices)
        return jnp.sum(residuals)

    def y0(self):
        # Initial values from SIF file
        return jnp.array([0.0, 10.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is approximately (1, 10, 1)
        return jnp.array([1.0, 10.0, 1.0])

    def expected_objective_value(self):
        # According to the SIF file comment (line 104),
        # the optimal objective value is 0.0
        return jnp.array(0.0)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOXBODLS(AbstractUnconstrainedMinimisation, strict=True):
    """NIST Data fitting problem BOXBOD.

    Fit: y = b1*(1-exp[-b2*x]) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Box, G. P., W. G. Hunter, and J. S. Hunter (1978).
    Statistics for Experimenters, New York, NY: Wiley, pp. 483-487.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: SUR2-MN-2-0
    """

    n: int = 2  # Problem has 2 variables
    m: int = 6  # Number of data points

    def objective(self, y, args):
        del args
        b1, b2 = y

        # Data points from the SIF file
        x_data = jnp.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        y_data = jnp.array([109.0, 149.0, 149.0, 191.0, 213.0, 224.0])

        # Model: y = b1*(1-exp[-b2*x])
        y_pred = b1 * (1.0 - jnp.exp(-b2 * x_data))

        # Compute the residuals
        residuals = y_pred - y_data

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([1.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # NIST certified values from https://www.itl.nist.gov/div898/strd/nls/data/boxbod.shtml
        return jnp.array([213.80937, 0.54723])

    def expected_objective_value(self):
        # From NIST: sum of squared residuals is 1.1680088766E+03
        return jnp.array(1.1680088766e03)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BOXPOWER(AbstractUnconstrainedMinimisation, strict=True):
    """Function with a "box-shaped" singular Hessian.

    Source: Nick Gould, June 2013

    Classification: OUR2-AN-V-0
    """

    def __init__(self, n=10000, p=9):
        """Initialize the BOXPOWER problem.

        Args:
            n: Dimension of the problem. Original value is 10000,
               other values are 10, 100, 1000, and 20000.
            p: Singularity type p leading to term x^2(p+1). Default is 9.
        """
        self.n = n
        self.p = p

    def objective(self, y, args):
        del args
        n = self.n
        p = self.p

        # Compute sum of squared terms for i=1 to n-1
        # Each term is (x[0] + x[i] + x[n-1])^2
        squared_terms = 0.0
        for i in range(1, n - 1):
            squared_terms += (y[0] + y[i] + y[n - 1]) ** 2

        # Add the first term (x[0])^2
        squared_terms += y[0] ** 2

        # Add the last term (x[n-1])^p
        power_term = y[n - 1] ** p

        return squared_terms + power_term

    def y0(self):
        # Initial values from SIF file (all 0.99)
        return jnp.full(self.n, 0.99)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 86),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
