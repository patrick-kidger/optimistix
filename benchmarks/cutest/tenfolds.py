import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class TENFOLDTRLS(AbstractUnconstrainedMinimisation, strict=True):
    """The 10FOLDTRLS function.

    The ten-fold triangular system whose root at zero has multiplicity 10.

    Source: Problem 8.3 in
    Wenrui Hao, Andrew J. Sommese and Zhonggang Zeng,
    "An algorithm and software for computing multiplicity structures
     at zeros of nonlinear systems", Technical Report,
    Department of Applied & Computational Mathematics & Statistics,
    University of Notre Dame, Indiana, USA (2012)

    SIF input: Nick Gould, Jan 2012.
    Least-squares version of 10FOLDTR.SIF, Nick Gould, Jun 2024.

    Classification: SUR2-AN-V-0
    """

    n: int = 1000  # Problem dimension, SIF file suggests 4, 10, 100, or 1000

    def objective(self, y, args):
        del args

        # For each i, compute the sum of elements from x_1 to x_i
        # This implements the triangular system described in the SIF file
        # Where E(i) = sum_{j=1}^i x_j for i=1...n
        e = jnp.cumsum(y)

        # Compute (E_{n-1})^4 and (E_n)^10 as described in the SIF file
        f1 = e[self.n - 2] ** 4  # E_{n-1}^4
        f2 = e[self.n - 1] ** 10  # E_n^10

        return f1 + f2

    def y0(self):
        # Initial value of 10.0 as specified in the SIF file
        return jnp.full(self.n, 10.0)

    def args(self):
        return None

    def expected_result(self):
        # The solution is at zero as mentioned in the SIF file description
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        return jnp.array(0.0)
