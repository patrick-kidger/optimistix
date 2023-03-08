import jax.flatten_util as jfu
import jax.scipy as jsp

from ..linear_operator import is_negative_semidefinite, is_positive_semidefinite
from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS


class Cholesky(AbstractLinearSolver):
    normal: bool = False

    def init(self, operator, options):
        del options
        matrix = operator.as_matrix()
        is_nsd = is_negative_semidefinite(operator)
        if self.normal:
            factor, lower = jsp.linalg.cho_factor(matrix.T @ matrix)
            state = (matrix, factor)
        else:

            if is_nsd:
                matrix = -matrix

            m, n = matrix.shape
            if m != n:
                raise ValueError(
                    "`Cholesky(..., normal=False)` may only be used for linear solves "
                    "with square matrices"
                )
            if not (is_positive_semidefinite(operator) | is_nsd):
                raise ValueError(
                    "`Cholesky(..., normal=False)` may only be used for positive "
                    "or negative definite linear operators"
                )

            state, lower = jsp.linalg.cho_factor(matrix)

        # Fix lower triangular, so that the boolean flag doesn't get needlessly promoted
        # to a tracer anywhere.
        assert lower is False
        return state, is_nsd

    def compute(self, state, vector, options):
        state, is_nsd = state
        del options
        vector, unflatten = jfu.ravel_pytree(vector)
        if self.normal:
            matrix, factor = state
            vector = matrix.T @ vector
        else:
            factor = state
        solution = jsp.linalg.cho_solve((factor, False), vector)
        if is_nsd and not self.normal:
            solution = -solution
        solution = unflatten(solution)
        return solution, RESULTS.successful, {}

    def pseudoinverse(self, operator):
        return False

    def transpose(self, state, options):
        if self.normal:
            # TODO(kidger): is there a way to compute this from the Cholesky
            # factorisation directly?
            matrix, _ = state
            m, _ = matrix.shape
            transpose_factor, lower = jsp.linalg.cho_factor(matrix @ matrix.T)
            assert lower is False
            transpose_state = matrix.T, transpose_factor
            transpose_options = {}
            return transpose_state, transpose_options
        else:
            # Matrix is symmetric anyway
            return state, options
