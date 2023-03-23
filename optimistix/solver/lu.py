import jax.scipy as jsp

from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS
from .misc import (
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


class LU(AbstractLinearSolver):
    """LU solver for linear systems.

    This solver can only handle square nonsingular operators.
    """

    def init(self, operator, options):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`LU` may only be used for linear solves with square matrices"
            )
        packed_structures = pack_structures(operator)
        return jsp.linalg.lu_factor(operator.as_matrix()), packed_structures, False

    def compute(self, state, vector, options):
        del options
        lu_and_piv, packed_structures, transpose = state
        trans = 1 if transpose else 0
        vector = ravel_vector(vector, packed_structures)
        solution = jsp.linalg.lu_solve(lu_and_piv, vector, trans=trans)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state, options):
        lu_and_piv, packed_structures, transpose = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = lu_and_piv, transposed_packed_structures, not transpose
        transpose_options = {}
        return transpose_state, transpose_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, opterator):
        return False
