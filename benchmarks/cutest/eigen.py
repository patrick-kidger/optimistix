import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: requires human review
class EIGEN(AbstractUnconstrainedMinimisation):
    """Base class for EIGEN problems.

    These problems compute eigenvalues and eigenvectors of symmetric matrices
    by solving a nonlinear least squares problem. They are formulated to find
    an orthogonal matrix Q and diagonal matrix D such that A = QᵀDQ where A
    is a specific input matrix.

    Different problems use different matrices A.

    Source: Originating from T.F. Coleman and P.A. Liao
    """

    n: int = 50  # Default dimension - individual problems may override

    def _matrix(self):
        """Return the specific matrix A for this eigenvalue problem.

        This method is implemented by each subclass for its specific matrix.
        """
        raise NotImplementedError("Subclasses must implement _matrix")

    def objective(self, y, args):
        del args

        # y contains the elements of Q and D packed together
        # First n² elements are Q in column-major order
        # Last n elements are the diagonal of D
        q_flat = y[: self.n * self.n]
        d_diag = y[self.n * self.n :]

        # Reshape Q to a matrix
        q = q_flat.reshape((self.n, self.n), order="F")

        # Create diagonal matrix D
        d = jnp.diag(d_diag)

        # Get the target matrix A
        a = self._matrix()

        # Calculate QᵀDQ
        qdq = q.T @ d @ q

        # Calculate residuals
        residuals = qdq - a

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Initialize with identity matrix for Q and zeros for diagonal of D
        q_flat = jnp.eye(self.n).flatten(order="F")  # Flatten in column-major order
        d_diag = jnp.zeros(self.n)

        return jnp.concatenate([q_flat, d_diag])

    def args(self):
        return None

    def expected_result(self):
        # The exact solution is not provided in the SIF files
        return None

    def expected_objective_value(self):
        # These problems should have a minimum of 0.0
        return jnp.array(0.0)


# TODO: requires human review
class EIGENALS(EIGEN):
    """EIGENALS - Eigenvalues of matrix A.

    Find eigenvalues and eigenvectors of a diagonal matrix A
    where the diagonal elements are 1, 2, ..., n.

    Source: Problem 57 in
    T.F. Coleman and P.A. Liao,
    "An efficient trust region method for unconstrained discrete-time
    optimal control problems",
    Computational Optimization and Applications 4(1), pp.47-66, 1995.

    SIF input: Nick Gould, Dec 1995.

    Classification: SUR2-AN-V-0
    """

    def _matrix(self):
        # Matrix A is diagonal with entries 1, 2, ..., n
        return jnp.diag(jnp.arange(1, self.n + 1))


# TODO: requires human review
class EIGENBLS(EIGEN):
    """EIGENBLS - Eigenvalues of matrix B.

    Find eigenvalues and eigenvectors of a tridiagonal matrix B
    with 2 on the main diagonal and -1 on the off-diagonals.

    Source: Problem 57 in
    T.F. Coleman and P.A. Liao,
    "An efficient trust region method for unconstrained discrete-time
    optimal control problems",
    Computational Optimization and Applications 4(1), pp.47-66, 1995.

    SIF input: Nick Gould, Dec 1995.

    Classification: SUR2-AN-V-0
    """

    def _matrix(self):
        # Matrix B is tridiagonal with 2 on diagonal and -1 on off-diagonals
        diag = jnp.full(self.n, 2.0)
        off_diag = jnp.full(self.n - 1, -1.0)

        # Create the tridiagonal matrix
        matrix = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

        return matrix


# TODO: requires human review
class EIGENCLS(EIGEN):
    """EIGENCLS - Eigenvalues of matrix C.

    Find eigenvalues and eigenvectors of a Wilkinson tridiagonal matrix C
    of size 2m+1, where m is a parameter.

    Source: Problem 57 in
    T.F. Coleman and P.A. Liao,
    "An efficient trust region method for unconstrained discrete-time
    optimal control problems",
    Computational Optimization and Applications 4(1), pp.47-66, 1995.

    SIF input: Nick Gould, Dec 1995.

    Classification: SUR2-AN-V-0
    """

    m: int = 25  # Half dimension minus 1
    n: int = 51  # Override base class n with 2*m+1

    def _matrix(self):
        # Matrix C is a Wilkinson tridiagonal matrix of size 2m+1
        # Diagonal entries are abs(i-m-1) for i=1...2m+1
        # Off-diagonal entries are 1

        # Create the diagonal
        indices = jnp.arange(1, self.n + 1)
        diag_values = jnp.abs(indices - self.m - 1)

        # Create the off-diagonals
        off_diag = jnp.ones(self.n - 1)

        # Create the tridiagonal matrix
        matrix = (
            jnp.diag(diag_values) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
        )

        return matrix
