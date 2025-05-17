import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROYDN3DLS(AbstractUnconstrainedMinimisation):
    """Broyden tridiagonal system of nonlinear equations in the least square sense.

    Source: problem 30 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Toint#17 and Buckley#78.
    SIF input: Ph. Toint, Dec 1989.
    Least-squares version: Nick Gould, Oct 2015.

    Classification: SUR2-AN-V-0
    """

    n: int = 5000  # Dimension of the problem
    kappa1: float = 2.0  # Parameter
    kappa2: float = 1.0  # Parameter

    def objective(self, y, args):
        del args
        n = self.n
        k1 = self.kappa1
        k2 = self.kappa2

        # Define function to compute residual for a specific index
        def compute_residual(i):
            if i == 0:
                # First residual: (3-2*x1)*x1 - 2*x2 + k2
                return (3.0 - k1 * y[0]) * y[0] - 2.0 * y[1] + k2
            elif i == n - 1:
                # Last residual: (3-2*xn)*xn - xn-1 + k2
                return (3.0 - k1 * y[n - 1]) * y[n - 1] - y[n - 2] + k2
            else:
                # Middle residuals: (3-2*xi)*xi - xi-1 - 2*xi+1 + k2
                return (3.0 - k1 * y[i]) * y[i] - y[i - 1] - 2.0 * y[i + 1] + k2

        # Create indices array
        indices = jnp.arange(n)

        # Compute residuals for all indices using vmap
        residuals = jax.vmap(compute_residual)(indices)

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return jnp.full(self.n, -1.0)

    def args(self):
        return None

    def expected_result(self):
        # Set values of all components to the same value r
        # where r is approximately -k2/(n*k1)
        k2 = self.kappa2
        n = self.n
        k1 = self.kappa1
        r = -k2 / (n * k1)
        return jnp.full(self.n, r)

    def expected_objective_value(self):
        # According to the SIF file comment (line 110),
        # the optimal objective value is 0.0
        return jnp.array(0.0)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROYDN7D(AbstractUnconstrainedMinimisation):
    """A seven diagonal variant of the Broyden tridiagonal system.

    Features a band far away from the diagonal.

    Source: Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#84
    SIF input: Ph. Toint, Dec 1989.

    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # Dimension of the problem (should be even)

    def objective(self, y, args):
        del args
        n = self.n
        half_n = n // 2

        # Function to compute the tridiagonal terms
        def compute_tridiagonal_term(i):
            if i == 0:
                # First term: (3-2*x1)*x1 - 2*x2 + 1
                return (3.0 - 2.0 * y[0]) * y[0] - 2.0 * y[1] + 1.0
            elif i == n - 1:
                # Last term: (3-2*xn)*xn - xn-1 + 1
                return (3.0 - 2.0 * y[n - 1]) * y[n - 1] - y[n - 2] + 1.0
            else:
                # Middle terms: (3-2*xi)*xi - xi-1 - 2*xi+1 + 1
                return (3.0 - 2.0 * y[i]) * y[i] - y[i - 1] - 2.0 * y[i + 1] + 1.0

        # Compute all tridiagonal terms using vmap
        indices = jnp.arange(n)
        tridiag_terms = jax.vmap(compute_tridiagonal_term)(indices)

        # Compute the additional terms from the distant band
        # For each i from 0 to n/2-1, add x_i + x_{i+n/2}
        band_terms = jnp.zeros(n)
        band_terms = band_terms.at[:half_n].set(y[:half_n] + y[half_n:])

        # Combine to get all terms
        all_terms = tridiag_terms + band_terms

        # Compute sum of |term_i|^(7/3) for all terms
        return jnp.sum(jnp.abs(all_terms) ** (7.0 / 3.0))

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 111),
        # the optimal objective value is 1.2701
        return jnp.array(1.2701)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BROYDNBDLS(AbstractUnconstrainedMinimisation):
    """Broyden banded system of nonlinear equations in least square sense.

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 and Toint#18
    SIF input: Ph. Toint, Dec 1989.
    Least-squares version: Nick Gould, Oct 2015

    Classification: SUR2-AN-V-0
    """

    n: int = 5000  # Dimension of the problem
    kappa1: float = 2.0  # Parameter
    kappa2: float = 5.0  # Parameter
    kappa3: float = 1.0  # Parameter
    lb: int = 5  # Lower bandwidth
    ub: int = 1  # Upper bandwidth

    def objective(self, y, args):
        del args
        n = self.n
        k1 = self.kappa1
        k2 = self.kappa2
        k3 = self.kappa3
        lb = self.lb
        ub = self.ub

        # Define compute_residual function for different regions
        def compute_upper_left_residual(i):
            # For indices 0 to lb-1
            term = (
                k1 * y[i]
                - k3 * jnp.sum(y[:i])
                - k3 * jnp.sum(y[i + 1 : jnp.minimum(i + ub + 1, n)])
            )
            return term - k2 * y[i] ** 3

        def compute_middle_residual(i):
            # For indices lb to n-ub-1
            term = (
                k1 * y[i]
                - k3 * jnp.sum(y[i - lb : i])
                - k3 * jnp.sum(y[i + 1 : i + ub + 1])
            )
            return term - k2 * y[i] ** 2

        def compute_lower_right_residual(i):
            # For indices n-ub to n-1
            term = k1 * y[i] - k3 * jnp.sum(y[i - lb : i]) - k3 * jnp.sum(y[i + 1 : n])
            return term - k2 * y[i] ** 3

        # Apply vmap to each region
        upper_indices = jnp.arange(lb)
        middle_indices = jnp.arange(lb, n - ub)
        lower_indices = jnp.arange(n - ub, n)

        upper_residuals = jax.vmap(compute_upper_left_residual)(upper_indices)
        middle_residuals = jax.vmap(compute_middle_residual)(middle_indices)
        lower_residuals = jax.vmap(compute_lower_right_residual)(lower_indices)

        # Combine all residuals
        all_residuals = jnp.concatenate(
            [upper_residuals, middle_residuals, lower_residuals]
        )

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 212),
        # the optimal objective value is 0.0
        return jnp.array(0.0)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class BRYBND(AbstractUnconstrainedMinimisation):
    """Broyden banded system of nonlinear equations.

    Source: problem 31 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    See also Buckley#73 (p. 41) and Toint#18
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 5000  # Dimension of the problem
    kappa1: float = 2.0  # Parameter
    kappa2: float = 5.0  # Parameter
    kappa3: float = 1.0  # Parameter
    lb: int = 5  # Lower bandwidth
    ub: int = 1  # Upper bandwidth

    def objective(self, y, args):
        del args
        n = self.n
        k1 = self.kappa1
        k2 = self.kappa2
        k3 = self.kappa3
        lb = self.lb
        ub = self.ub

        # Define compute_residual function for different regions
        def compute_upper_left_residual(i):
            # For indices 0 to lb-1
            term = (
                k1 * y[i]
                - k3 * jnp.sum(y[:i])
                - k3 * jnp.sum(y[i + 1 : jnp.minimum(i + ub + 1, n)])
            )
            return term - k2 * y[i] ** 3

        def compute_middle_residual(i):
            # For indices lb to n-ub-1
            term = (
                k1 * y[i]
                - k3 * jnp.sum(y[i - lb : i])
                - k3 * jnp.sum(y[i + 1 : i + ub + 1])
            )
            return term - k2 * y[i] ** 2

        def compute_lower_right_residual(i):
            # For indices n-ub to n-1
            term = k1 * y[i] - k3 * jnp.sum(y[i - lb : i]) - k3 * jnp.sum(y[i + 1 : n])
            return term - k2 * y[i] ** 3

        # Apply vmap to each region
        upper_indices = jnp.arange(lb)
        middle_indices = jnp.arange(lb, n - ub)
        lower_indices = jnp.arange(n - ub, n)

        upper_residuals = jax.vmap(compute_upper_left_residual)(upper_indices)
        middle_residuals = jax.vmap(compute_middle_residual)(middle_indices)
        lower_residuals = jax.vmap(compute_lower_right_residual)(lower_indices)

        # Combine all residuals
        all_residuals = jnp.concatenate(
            [upper_residuals, middle_residuals, lower_residuals]
        )

        # Return the sum of squared residuals
        return jnp.sum(all_residuals**2)

    def y0(self):
        # Initial values from SIF file (all 1.0)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        return None

    def expected_objective_value(self):
        # According to the SIF file comment (line 213),
        # the optimal objective value is 0.0
        return jnp.array(0.0)
