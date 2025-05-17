import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROWNAL(AbstractUnconstrainedMinimisation, strict=True):
    """Brown almost linear least squares problem.

    This problem is a sum of n least-squares groups, the last one of
    which has a nonlinear element. Its Hessian matrix is dense.

    Source: Problem 27 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#79
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    def __init__(self, n=200):
        """Initialize the BROWNAL problem.

        Args:
            n: Dimension of the problem. Original value is 10,
               other values are 100, 200, and 1000.
        """
        self.n = n

    def objective(self, y, args):
        del args
        n = self.n

        # Sum of squared residuals
        residuals = []

        # First n-1 groups: sum of components - (n+1)
        for i in range(n - 1):
            residual = jnp.sum(y) - (n + 1)
            residuals.append(residual)

        # Last group: product of components - 1
        product = jnp.prod(y)
        residuals.append(product - 1.0)

        return jnp.sum(jnp.array(residuals) ** 2)

    def y0(self):
        # Initial values from SIF file (all 0.5)
        return jnp.full(self.n, 0.5)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution consists of all 1s except the last component
        # which is 1/(n-1)
        n = self.n
        result = jnp.ones(n)
        result = result.at[n - 1].set(1.0 / (n - 1))
        return result

    def expected_objective_value(self):
        # According to the SIF file comment (line 121),
        # the optimal objective value is 0.0
        return jnp.array(0.0)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROWNBS(AbstractUnconstrainedMinimisation, strict=True):
    """Brown and Dennis function.

    Source: Problem 16 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#17
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-4-0
    """

    n: int = 4  # Problem has 4 variables
    m: int = 20  # Number of data points

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Define grid points
        t_values = jnp.array([i / 5 for i in range(1, self.m + 1)])

        # Define the residual function
        def compute_residual(t):
            term1 = (x1 + t * x2 - jnp.exp(t)) ** 2
            term2 = (x3 + x4 * jnp.sin(t) - jnp.cos(t)) ** 2
            return term1 + term2

        # Compute all residuals using vmap
        residuals = jax.vmap(compute_residual)(t_values)

        return jnp.sum(residuals)

    def y0(self):
        # Initial values from literature
        return jnp.array([25.0, 5.0, -5.0, -1.0])

    def args(self):
        return None

    def expected_result(self):
        # The solution is approximately
        # (x1,x2,x3,x4) = (-11.59..., 13.20..., -0.40..., 0.24...)
        return jnp.array([-11.594, 13.203, -0.4034, 0.2367])

    def expected_objective_value(self):
        # The optimal objective value is approximately 85822.2
        return jnp.array(85822.2)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROWNDEN(AbstractUnconstrainedMinimisation, strict=True):
    """Brown badly scaled function.

    Source: Problem 4 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#36
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-4-0
    """

    n: int = 4  # Problem has 4 variables
    m: int = 20  # Number of data points

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Create vector of t values from 1 to 20
        t_values = jnp.arange(1, self.m + 1)

        # Compute terms for each t value
        def compute_residual(t):
            numerator = x1 + x2 * t - jnp.exp(t)
            denominator = x3 + x4 * jnp.exp(t / 2) - jnp.exp(t / 2)
            return numerator / denominator

        # Vectorize the computation over all t values
        residuals = jax.vmap(compute_residual)(t_values)

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from literature
        return jnp.array([25.0, 5.0, -5.0, -1.0])

    def args(self):
        return None

    def expected_result(self):
        # The solution approximately (1000, 10^-3, 1, 100)
        return jnp.array([1000.0, 1.0e-3, 1.0, 100.0])

    def expected_objective_value(self):
        # The optimal objective value is approximately 0.0
        return jnp.array(0.0)
