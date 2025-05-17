import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed to verify the implementation matches the problem definition
class INTEQNELS(AbstractUnconstrainedMinimisation, strict=True):
    """The discrete integral problem (INTEGREQ) in least-squares form.

    Source: Problem 29 in
    J.J. More', B.S. Garbow and K.E. Hillstrom,
    "Testing Unconstrained Optimization Software",
    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.

    SIF input: Ph. Toint, Feb 1990.
    Modification to remove fixed variables: Nick Gould, Oct 2015.
    Classification: SUR2-AN-V-0
    """

    n: int = 500  # Other suggested values: 10, 50, 100

    def objective(self, y, args):
        """Compute the objective function value.

        This problem involves a system of integral equations discretized on N+2 points.
        The objective is the sum of squares of differences between the right-hand side
        and the left-hand side of the discretized integral equation.
        """
        n = self.n
        h = 1.0 / (n + 1)
        half_h = 0.5 * h

        # Create full variable array with fixed boundary values
        # (x[0] and x[n+1] are fixed at 0)
        x_full = jnp.zeros(n + 2)
        x_full = x_full.at[1:-1].set(y)  # Insert free variables

        # Precompute t values (grid points)
        t = jnp.arange(n + 2) * h

        # Define the cube term function
        def cube_term_fn(j):
            return (x_full[j] + 1.0 + t[j]) ** 3

        # Vectorize the cube term computation
        cube_terms = jax.vmap(cube_term_fn)(jnp.arange(1, n + 1))

        # Define the computation for a single residual at point i
        def residual_fn(i):
            ti = t[i]

            # Define the lower part weight function (for j=1 to i)
            def lower_weight_fn(j):
                tj = t[j]
                return (1.0 - ti) * half_h * tj

            # Define the upper part weight function (for j=i+1 to n)
            def upper_weight_fn(j):
                tj = t[j]
                return ti * half_h * (1.0 - tj)

            # Apply weights to the cube terms using vectorized operations
            lower_indices = jnp.arange(1, i + 1)
            upper_indices = jnp.arange(i + 1, n + 1)

            # Compute weighted sums using vmap
            lower_weights = jax.vmap(lower_weight_fn)(lower_indices)
            upper_weights = jax.vmap(upper_weight_fn)(upper_indices)

            lower_sum = jnp.sum(lower_weights * cube_terms[lower_indices - 1])
            upper_sum = jnp.sum(upper_weights * cube_terms[upper_indices - 1])

            # Compute residual
            return x_full[i] - lower_sum - upper_sum

        # Compute all residuals
        residuals = jax.vmap(residual_fn)(jnp.arange(1, n + 1))

        return jnp.sum(residuals**2)

    def y0(self):
        """Initial point: x_i = t_i * (t_i - 1) for i=1,...,n."""
        n = self.n
        h = 1.0 / (n + 1)
        t = jnp.arange(1, n + 1) * h
        return t * (t - 1.0)

    def args(self):
        """No additional arguments needed."""
        return None

    def expected_result(self):
        """The solution is not specified in the SIF file."""
        return None

    def expected_objective_value(self):
        """The solution value is not specified in the SIF file."""
        return None
