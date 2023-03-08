from typing import Optional

import jax.numpy as jnp

from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS
from .misc import (
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)
from .triangular import solve_triangular


class QR(AbstractLinearSolver):
    """QR solver for linear systems.

    This solver can handle singular operators.

    This is usually the preferred solver when dealing with singular operators.

    Equivalent to `scipy.linalg.lstsq`.
    """

    rcond: Optional[float] = None

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
            # Minimal norm solution if ill-posed
            solution = q @ solve_triangular(
                r.T, vector, lower=True, unit_diagonal=False, rcond=self.rcond
            )
        else:
            # Least squares solution if ill-posed
            solution = solve_triangular(
                r, q.T @ vector, lower=False, unit_diagonal=False, rcond=self.rcond
            )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def pseudoinverse(self, operator):
        return True

    def transpose(self, state, options):
        (q, r), transpose, structures = state
        transposed_packed_structures = transpose_packed_structures(structures)
        transpose_state = (q, r), not transpose, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options


QR.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator. (I.e.
    `N` is the output size and `M` is the input size.)
"""
