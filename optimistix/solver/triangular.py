import jax.scipy as jsp

from ..linear_operator import (
    has_unit_diagonal,
    is_lower_triangular,
    is_upper_triangular,
)
from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS
from .misc import (
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


class Triangular(AbstractLinearSolver):
    """Triangular solver for linear systems.

    The operator should either be lower triangular or upper triangular.
    """

    def init(self, operator, options):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Triangular` may only be used for linear solves with square matrices"
            )
        if not (is_lower_triangular(operator) or is_upper_triangular(operator)):
            raise ValueError(
                "`Triangular` may only be used for linear solves with triangular "
                "matrices"
            )
        return (
            operator.as_matrix(),
            is_lower_triangular(operator),
            has_unit_diagonal(operator),
            pack_structures(operator),
            False,  # transposed
        )

    def compute(self, state, vector, options):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        del state, options
        vector = ravel_vector(vector, packed_structures)
        if transpose:
            trans = "T"
        else:
            trans = "N"
        solution = jsp.linalg.solve_triangular(
            matrix, vector, trans=trans, lower=lower, unit_diagonal=unit_diagonal
        )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state, options):
        matrix, lower, unit_diagonal, packed_structures, transpose = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (
            matrix,
            lower,
            unit_diagonal,
            transposed_packed_structures,
            not transpose,
        )
        transpose_options = {}
        return transpose_state, transpose_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False
