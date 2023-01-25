import jax.flatten_util as jfu
import jax.scipy as jsp

from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS


class LU(AbstractLinearSolver):
    def init(self, operator, options):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`LU` may only be used for linear solves with square matrices"
            )
        return jsp.linalg.lu_factor(operator.as_matrix()), False

    def compute(self, state, vector, options):
        del options
        lu_and_piv, transpose = state
        trans = 1 if transpose else 0
        vector, unflatten = jfu.ravel_pytree(vector)
        solution = unflatten(jsp.linalg.lu_solve(lu_and_piv, vector, trans=trans))
        return solution, RESULTS.successful, {}

    def pseudoinverse(self, operator):
        return False

    def transpose(self, state, options):
        lu_and_piv, transpose = state
        transpose_state = lu_and_piv, not transpose
        transpose_options = {}
        return transpose_state, transpose_options
