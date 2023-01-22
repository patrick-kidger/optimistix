from typing import Optional

import jax.flatten_util as jfu
import jax.numpy as jnp

from ..linear_solve import AbstractLinearSolver
from ..misc import resolve_rcond
from ..solution import RESULTS


class Diagonal(AbstractLinearSolver):
    maybe_singular: bool = True
    rcond: Optional[float] = None

    def is_maybe_singular(self):
        return self.maybe_singular

    def will_materialise(self, operator):
        return True

    def init(self, operator, options):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Diagonal` may only be used for linear solves with square matrices"
            )
        if not operator.pattern.diagonal:
            raise ValueError(
                "`Diagonal` may only be used for linear solves with diagonal matrices"
            )
        return operator

    def compute(self, state, vector, options):
        operator = state
        del state, options
        if operator.pattern.unit_diagonal:
            solution = vector
        else:
            # TODO(kidger): do diagonal solves more efficiently than this.
            vector, unflatten = jfu.ravel_pytree(vector)
            diag = jnp.diag(operator.as_matrix())
            rcond = resolve_rcond(self.rcond, diag.size, diag.size, diag.dtype)
            diag = jnp.where(jnp.abs(diag) >= rcond * jnp.max(diag), diag, jnp.inf)
            solution = unflatten(vector / diag)
        return solution, RESULTS.successful, {}

    def transpose(self, state, options):
        # Matrix is symmetric
        return state, options
