import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CHNROSNB(AbstractUnconstrainedMinimisation):
    """The chained Rosenbrock function (Toint).

    This is a variant of the Rosenbrock function where each variable is connected to
    its neighbors in a chain-like structure. The variables in each pair (x_{i-1}, x_i)
    are scaled by a parameter alpha_i.

    Source:
    Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#46 (n = 25) (p. 45).
    SIF input: Ph. Toint, Dec 1989.

    Classification: SUR2-AN-V-0
    """

    n: int = 50  # Number of variables (default 50, but can also be 10 or 25)

    def objective(self, y, args):
        del args

        # Alpha values from the SIF file
        alphas = jnp.array(
            [
                1.25,
                1.40,
                2.40,
                1.40,
                1.75,
                1.20,
                2.25,
                1.20,
                1.00,
                1.10,
                1.50,
                1.60,
                1.25,
                1.25,
                1.20,
                1.20,
                1.40,
                0.50,
                0.50,
                1.25,
                1.80,
                0.75,
                1.25,
                1.40,
                1.60,
                2.00,
                1.00,
                1.60,
                1.25,
                2.75,
                1.25,
                1.25,
                1.25,
                3.00,
                1.50,
                2.00,
                1.25,
                1.40,
                1.80,
                1.50,
                2.20,
                1.40,
                1.50,
                1.25,
                2.00,
                1.50,
                1.25,
                1.40,
                0.60,
                1.50,
            ]
        )

        # Ensure we're using the correct number of alpha values
        # based on problem dimension
        alphas = alphas[: self.n]

        # Define the function to compute each term for a pair of variables
        def compute_term(i):
            # Each term in the objective is of the form:
            # 16 * alpha_i^2 * (x_i - x_{i-1}^2)^2
            alpha_i = alphas[i - 1]  # Alpha is 1-indexed in the SIF
            scale = 16.0 * alpha_i * alpha_i
            term = scale * (y[i] - y[i - 1] ** 2) ** 2
            return term

        # Create array of indices (2 to n)
        indices = jnp.arange(1, self.n)

        # Compute all terms using vmap
        terms = jax.vmap(compute_term)(indices)

        # Sum all terms
        return jnp.sum(terms)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return jnp.full(self.n, -1.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # According to the SIF file comment, the optimal objective value is 0.0
        return jnp.array(0.0)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CHNRSNBM(AbstractUnconstrainedMinimisation):
    """A variable dimension version of the chained Rosenbrock function (CHNROSNB)
    by Luksan et al.

    This is a modification of the CHNROSNB function where the alpha values are
    determined by the formula: alpha_i = sin(i) * 1.5

    Source: problem 27 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    that is an extension of that proposed in
    Ph.L. Toint,
    "Some numerical results using a sparse matrix updating formula in
    unconstrained optimization",
    Mathematics of Computation, vol. 32(114), pp. 839-852, 1978.

    See also Buckley#46 (n = 25) (p. 45).
    SIF input: Ph. Toint, Dec 1989.
              this version Nick Gould, June, 2013

    Classification: SUR2-AN-V-0
    """

    n: int = 50  # Number of variables (default 50, but can also be 10 or 25)

    def objective(self, y, args):
        del args

        # Define the function to compute each term for a pair of variables
        def compute_term(i):
            # Alpha values are determined by sin(i) * 1.5
            alpha_i = jnp.sin(float(i)) * 1.5

            # Each term in the objective is of the form:
            # 16 * alpha_i^2 * (x_i - x_{i-1}^2)^2
            scale = 16.0 * alpha_i * alpha_i
            term = scale * (y[i] - y[i - 1] ** 2) ** 2
            return term

        # Create array of indices (2 to n)
        indices = jnp.arange(1, self.n)

        # Compute all terms using vmap
        terms = jax.vmap(compute_term)(indices)

        # Sum all terms
        return jnp.sum(terms)

    def y0(self):
        # Initial values from SIF file (all -1.0)
        return jnp.full(self.n, -1.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # According to the SIF file comment, the optimal objective value is 0.0
        return jnp.array(0.0)
