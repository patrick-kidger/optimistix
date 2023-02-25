from typing import Optional

import jax.flatten_util as jfu
import jax.numpy as jnp

from ..linear_operator import diagonal, has_unit_diagonal, is_diagonal
from ..linear_solve import AbstractLinearSolver
from ..misc import resolve_rcond
from ..solution import RESULTS


class Diagonal(AbstractLinearSolver):
    rcond: Optional[float] = None

    def init(self, operator, options):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Diagonal` may only be used for linear solves with square matrices"
            )
        if not is_diagonal(operator):
            raise ValueError(
                "`Diagonal` may only be used for linear solves with diagonal matrices"
            )
        if has_unit_diagonal(operator):
            diag = None
        else:
            diag = diagonal(operator)
        return diag, has_unit_diagonal(operator)

    def compute(self, state, vector, options):
        diag, unit_diagonal = state
        del state, options
        if unit_diagonal:
            solution = vector
        else:
            vector, unflatten = jfu.ravel_pytree(vector)
            (size,) = diag.shape
            rcond = resolve_rcond(self.rcond, size, size, diag.dtype)
            diag = jnp.where(jnp.abs(diag) >= rcond * jnp.max(diag), diag, jnp.inf)
            solution = unflatten(vector / diag)
        return solution, RESULTS.successful, {}

    def pseudoinverse(self, operator):
        return True

    def transpose(self, state, options):
        # Matrix is symmetric
        return state, options
