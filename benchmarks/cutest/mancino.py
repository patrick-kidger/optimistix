import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar

from .problem import AbstractUnconstrainedMinimisation


# TODO human review
class MANCINO(AbstractUnconstrainedMinimisation):
    """Mancino's function with variable dimension.

    Source:
        E. Spedicato,
        "Computational experience with quasi-Newton algorithms for
        minimization problems of moderate size",
        Report N-175, CISE, Milano, 1975.

    See also:
        Buckley #51 (p. 72), Schittkowski #391 (for N = 30)

    SIF input:
        Ph. Toint, Dec 1989.
        correction by Ph. Shott, January, 1995.
        correction by S. Gratton & Ph. Toint, May 2024

    Classification: SUR2-AN-V-0
    """

    n: int = 100
    alpha: int = 5
    beta: float = 14.0
    gamma: int = 3

    def objective(self, y: Float[Array, " n"], args: PyTree) -> Scalar:
        """Compute the objective function for the Mancino problem."""
        n = self.n
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        # Precompute some constants
        n_minus_1 = n - 1
        n_minus_1_sq = n_minus_1 * n_minus_1
        beta_n = beta * n
        beta_n_sq = beta_n * beta_n
        alpha_plus_1 = alpha + 1
        alpha_plus_1_sq = alpha_plus_1 * alpha_plus_1
        f0 = alpha_plus_1_sq * n_minus_1_sq
        f1 = -f0
        f2 = beta_n_sq + f1
        f3 = f2 / 1.0
        f4 = beta_n * f3
        a = -f4

        # Function to compute a term for a given (i,j) pair
        def compute_term(i, j):
            i_float = i
            j_float = j
            i_over_j = i_float / j_float
            sqrt_i_over_j = jnp.sqrt(i_over_j)
            log_value = jnp.log(sqrt_i_over_j)
            sin_value = jnp.sin(log_value)
            cos_value = jnp.cos(log_value)

            # Compute sin^α and cos^α
            sin_pow = jnp.power(sin_value, alpha)
            cos_pow = jnp.power(cos_value, alpha)

            # Return contribution
            return sqrt_i_over_j * (sin_pow + cos_pow)

        # Compute the coefficients C_i
        def compute_c_i(i):
            i_minus_half_n = i - n / 2
            return jnp.power(i_minus_half_n, gamma)

        # Function to compute the sum of terms for index i
        def compute_sum_for_index(i):
            # Create index arrays for j < i and j > i
            lower_js = jnp.arange(1, i, dtype=jnp.float32)
            upper_js = jnp.arange(i + 1, n + 1, dtype=jnp.float32)

            # Use vmap to compute terms for all j values
            i_repeated_lower = jnp.full_like(lower_js, i)
            i_repeated_upper = jnp.full_like(upper_js, i)

            lower_terms = jax.vmap(compute_term)(i_repeated_lower, lower_js)
            upper_terms = jax.vmap(compute_term)(i_repeated_upper, upper_js)

            # Return sum of all terms
            return jnp.sum(lower_terms) + jnp.sum(upper_terms)

        # Compute the objective function
        def compute_residual(i):
            # Convert to float for computation
            i_float = float(i)

            # Get coefficient C_i
            c_i = compute_c_i(i_float)

            # Compute sum of terms
            sum_terms = compute_sum_for_index(i_float)

            # Full term including C_i
            full_term = beta * y[i - 1] + a * (sum_terms + c_i)

            # Residual
            return c_i * full_term**2

        # Map the residual calculation over all indices
        indices = jnp.arange(1, n + 1, dtype=jnp.float32)
        residuals = jax.vmap(compute_residual)(indices)

        # Sum all residuals
        return jnp.sum(residuals)

    def y0(self) -> Float[Array, " n"]:
        """Initial guess for the Mancino problem."""
        n = self.n
        alpha = self.alpha
        gamma = self.gamma

        # Precompute constants for the starting point calculation
        n_minus_1 = n - 1
        n_minus_1_sq = n_minus_1 * n_minus_1
        beta_n = self.beta * n
        beta_n_sq = beta_n * beta_n
        alpha_plus_1 = alpha + 1
        alpha_plus_1_sq = alpha_plus_1 * alpha_plus_1
        f0 = alpha_plus_1_sq * n_minus_1_sq
        f1 = -f0
        f2 = beta_n_sq + f1
        f3 = f2 / 1.0
        f4 = beta_n * f3
        a = -f4

        # Function to compute a term for a given (i,j) pair
        def compute_term(i, j):
            i_float = i
            j_float = j
            i_over_j = i_float / j_float
            sqrt_i_over_j = jnp.sqrt(i_over_j)
            log_value = jnp.log(sqrt_i_over_j)
            sin_value = jnp.sin(log_value)
            cos_value = jnp.cos(log_value)

            # Compute sin^α and cos^α
            sin_pow = jnp.power(sin_value, alpha)
            cos_pow = jnp.power(cos_value, alpha)

            # Return contribution
            return sqrt_i_over_j * (sin_pow + cos_pow)

        # Function to compute initial value for a specific i
        def compute_x_i(i):
            # Convert to float
            i_float = float(i)

            # Compute C_i
            i_minus_half_n = i_float - n / 2
            c_i = jnp.power(i_minus_half_n, gamma)

            # Create index arrays for j < i and j > i
            lower_js = jnp.arange(1, i, dtype=jnp.float32)
            upper_js = jnp.arange(i + 1, n + 1, dtype=jnp.float32)

            # Use vmap to compute terms for all j values
            i_repeated_lower = jnp.full_like(lower_js, i_float)
            i_repeated_upper = jnp.full_like(upper_js, i_float)

            lower_terms = jax.vmap(compute_term)(i_repeated_lower, lower_js)
            upper_terms = jax.vmap(compute_term)(i_repeated_upper, upper_js)

            # Sum all terms
            sum_terms = jnp.sum(lower_terms) + jnp.sum(upper_terms)

            # Return initial value
            return a * (sum_terms + c_i)

        # Compute initial values for all indices
        indices = jnp.arange(1, n + 1)
        return jax.vmap(compute_x_i)(indices)

    def args(self) -> None:
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
