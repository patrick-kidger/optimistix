from typing import Optional

import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.scipy as jsp

from ..linear_solve import AbstractLinearSolver


class Cholesky(AbstractLinearSolver):
    maybe_singular: Optional[bool] = None
    normal: bool = False
    jitter: float = 0

    def is_maybe_singular(self):
        if self.maybe_singular is None:
            return self.normal and self.jitter > 0
        else:
            return self.maybe_singular

    def will_materialise(self):
        return True

    def init(self, operator, options):
        del options
        if self.normal:
            matrix = operator.as_matrix()
            _, n = matrix.shape
            factor, lower = jsp.linalg.cho_factor(
                matrix.T @ matrix + self.jitter * jnp.eye(n)
            )
            state = matrix, factor
        else:
            if operator.in_size() != operator.out_size():
                raise ValueError(
                    "`Cholesky(..., normal=False)` may only be used for linear solves "
                    "with square matrices"
                )
            if not operator.pattern.symmetric:
                raise ValueError(
                    "`Cholesky(..., normal=False)` may only be used for symmetric "
                    "linear operators"
                )
            state, lower = jsp.linalg.cho_factor(operator.as_matrix())
        # Fix lower triangular, so that the boolean flag doesn't get needlessly promoted
        # to a tracer anywhere.
        assert lower is False
        return state

    def solve(self, state, vector, options):
        del options
        vector, unflatten = jfu.ravel_pytree(vector)
        if self.normal:
            matrix, factor = state
            vector = matrix.T @ vector
        else:
            factor = state
        return unflatten(jsp.linalg.cho_solve((factor, False), vector))

    def transpose(self, state, options):
        if self.normal:
            # TODO(kidger): is there a way to compute this from the Cholesky
            # factorisation directly?
            matrix, _ = state
            m, _ = matrix.shape
            transpose_factor, lower = jsp.linalg.cho_factor(
                matrix @ matrix.T + self.jitter * jnp.eye(m)
            )
            assert lower is False
            transpose_state = matrix.T, transpose_factor
            transpose_options = {}
            return transpose_state, transpose_options
        else:
            # Matrix is symmetric anyway
            return state, options
