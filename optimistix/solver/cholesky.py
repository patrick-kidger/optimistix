import jax.flatten_util as jfu
import jax.scipy as jsp

from ..linear_operator import is_negative_semidefinite, is_positive_semidefinite
from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS


class Cholesky(AbstractLinearSolver):
    """Cholesky solver for linear systems. This is generally the preferred solver for
    positive or negative definite systems.

    Equivalent to `scipy.linalg.solve(..., assume_a="pos")`.

    The operator must be square, nonsingular, and either positive or negative definite.
    """

    def init(self, operator, options):
        del options
        is_nsd = is_negative_semidefinite(operator)
        if not (is_positive_semidefinite(operator) | is_nsd):
            raise ValueError(
                "`Cholesky(..., normal=False)` may only be used for positive "
                "or negative definite linear operators"
            )
        matrix = operator.as_matrix()
        m, n = matrix.shape
        if m != n:
            raise ValueError(
                "`Cholesky(..., normal=False)` may only be used for linear solves "
                "with square matrices"
            )
        if is_nsd:
            matrix = -matrix
        factor, lower = jsp.linalg.cho_factor(matrix)
        # Fix lower triangular for simplicity.
        assert lower is False
        return factor, is_nsd

    def compute(self, state, vector, options):
        factor, is_nsd = state
        del options
        # Cholesky => PSD => symmetric => (in_structure == out_structure) =>
        # we don't need to use packed structures.
        vector, unflatten = jfu.ravel_pytree(vector)
        solution = jsp.linalg.cho_solve((factor, False), vector)
        if is_nsd:
            solution = -solution
        solution = unflatten(solution)
        return solution, RESULTS.successful, {}

    def transpose(self, state, options):
        # Matrix is symmetric anyway
        return state, options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False
