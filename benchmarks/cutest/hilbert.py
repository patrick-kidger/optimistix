import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class HILBERTA(AbstractUnconstrainedMinimisation, strict=True):
    """Hilbert matrix problem.

    Unconstrained quadratic minimization problem using a Hilbert matrix.
    The Hilbert matrix is notorious for being badly conditioned, which makes
    this a challenging test problem for optimization algorithms.

    Source:
    K. Schittkowski,
    "More Test Examples for Nonlinear Programming Codes",
    Springer Verlag, Heidelberg, 1987.

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    n: int = 2  # Other suggested values: 4, 5, 6, 10, works for any positive integer

    def objective(self, y, args):
        del args

        # Diagonal perturbation parameter (from SIF file)
        d = 0.0

        # Create Hilbert matrix using the formula h_ij = 1/(i+j-1)
        # Use meshgrid to generate indices
        i_indices = jnp.arange(1, self.n + 1)
        j_indices = jnp.arange(1, self.n + 1)
        i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing="ij")

        # Create the Hilbert matrix
        hilbert_matrix = 1.0 / (i_grid + j_grid - 1.0)

        # Add the diagonal perturbation
        diag_indices = jnp.diag_indices(self.n)
        perturbed_matrix = hilbert_matrix.at[diag_indices].add(d)

        # Compute the quadratic form: 0.5 * y^T * H * y
        return 0.5 * jnp.dot(y, jnp.dot(perturbed_matrix, y))

    def y0(self):
        # Starting point: all variables set to -3.0 (from SIF file)
        return jnp.full(self.n, -3.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        # For a quadratic problem, the solution can be computed by solving Hx=0
        # but we'll leave this as None since the result depends on the dimension n
        return None

    def expected_objective_value(self):
        # The minimum value of the quadratic form is 0.0 (since it's positive definite)
        return jnp.array(0.0)


# TODO: human review required
class HILBERTB(AbstractUnconstrainedMinimisation, strict=True):
    """Perturbed Hilbert matrix problem.

    Unconstrained quadratic minimization problem using a Hilbert matrix
    with a diagonal perturbation to improve conditioning. The Hilbert matrix
    is notorious for being badly conditioned, and this perturbation makes
    the problem more tractable.

    Source: problem 19 (p. 59) in
    A.R. Buckley,
    "Test functions for unconstrained minimization",
    TR 1989CS-3, Mathematics, statistics and computing centre,
    Dalhousie University, Halifax (CDN), 1989.

    SIF input: Ph. Toint, Dec 1989.

    Classification: QUR2-AN-V-0
    """

    n: int = (
        10  # Other suggested values in SIF: 5, 50, should work for any positive integer
    )

    def objective(self, y, args):
        del args

        # Diagonal perturbation parameter - the key difference from HILBERTA
        # Value taken directly from the SIF file
        d = 5.0

        # Create Hilbert matrix using the formula h_ij = 1/(i+j-1)
        # Use meshgrid to generate indices
        i_indices = jnp.arange(1, self.n + 1)
        j_indices = jnp.arange(1, self.n + 1)
        i_grid, j_grid = jnp.meshgrid(i_indices, j_indices, indexing="ij")

        # Create the Hilbert matrix
        hilbert_matrix = 1.0 / (i_grid + j_grid - 1.0)

        # Add the diagonal perturbation
        diag_indices = jnp.diag_indices(self.n)
        perturbed_matrix = hilbert_matrix.at[diag_indices].add(d)

        # Compute the quadratic form: 0.5 * y^T * H * y
        return 0.5 * jnp.dot(y, jnp.dot(perturbed_matrix, y))

    def y0(self):
        # Starting point: all variables set to -3.0 (from SIF file)
        return jnp.full(self.n, -3.0)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not explicitly provided in the SIF file
        # For a quadratic problem, the solution can be computed by solving Hx=0
        # However, since H is non-singular (due to the diagonal perturbation),
        # the optimal solution is x = 0
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # The minimum value of the quadratic form is 0.0
        return jnp.array(0.0)
