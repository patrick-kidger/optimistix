import jax.numpy as jnp
import jax.scipy as jsp

from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS
from .misc import (
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


class QR(AbstractLinearSolver):
    """QR solver for linear systems.

    This solver can handle non-square operators.

    This is usually the preferred solver when dealing with non-square operators.

    !!! info

        Note that whilst this does handle non-square operators, it still cannot handle
        singular operators (i.e. operators with less than full rank).

        This is because JAX does not currently support a rank-revealing/pivoted QR
        decomposition, see [issue #12897](https://github.com/google/jax/issues/12897).

        For such use cases, switch to [`optimistix.SVD`][] instead.
    """

    def init(self, operator, options):
        del options
        matrix = operator.as_matrix()
        m, n = matrix.shape
        transpose = n > m
        if transpose:
            matrix = matrix.T
        qr = jnp.linalg.qr(matrix, mode="reduced")
        packed_structures = pack_structures(operator)
        return qr, transpose, packed_structures

    def compute(self, state, vector, options):
        (q, r), transpose, packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if transpose:
            # Minimal norm solution if underdetermined.
            solution = q @ jsp.linalg.solve_triangular(
                r, vector, trans="T", unit_diagonal=False
            )
        else:
            # Least squares solution if overdetermined.
            solution = jsp.linalg.solve_triangular(
                r, q.T @ vector, trans="N", unit_diagonal=False
            )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state, options):
        (q, r), transpose, structures = state
        transposed_packed_structures = transpose_packed_structures(structures)
        transpose_state = (q, r), not transpose, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def allow_dependent_columns(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        # We're able to pull an efficiency trick here.
        #
        # As we don't use a rank-revealing implementation, then we always require that
        # the operator have full rank.
        #
        # So if we have columns <= rows, then we know that all our columns are linearly
        # independent. We can return `False` and get a computationally cheaper jvp rule.
        return columns > rows

    def allow_dependent_rows(self, operator):
        rows = operator.out_size()
        columns = operator.in_size()
        return rows > columns
