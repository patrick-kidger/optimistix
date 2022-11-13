from typing import Optional

import equinox.internal as eqxi
import jax.flatten_util as jfu
import jax.numpy as jnp

from ..linear_solve import AbstractLinearSolver
from ..solution import RESULTS
from .triangular import solve_triangular


class QR(AbstractLinearSolver):
    maybe_singular: bool = True
    rcond: Optional[float] = None

    def is_maybe_singular(self):
        return self.maybe_singular

    def init(self, operator, options):
        del options
        matrix = operator.as_matrix()
        m, n = matrix.shape
        transpose = n > m
        if transpose:
            matrix = matrix.T
        qr = jnp.linalg.qr(matrix, mode="reduced")
        return qr, eqxi.Static(transpose)

    def compute(self, state, vector, options):
        (q, r), transpose = state
        del state, options
        vector, unflatten = jfu.ravel_pytree(vector)
        if transpose.value:
            # Minimal norm solution if ill-posed
            solution = q @ solve_triangular(
                r.T, vector, lower=False, unit_diagonal=False, rcond=self.rcond
            )
        else:
            # Least squares solution if ill-posed
            solution = solve_triangular(
                r, q.T @ vector, lower=False, unit_diagonal=False, rcond=self.rcond
            )
        solution = unflatten(solution)
        return solution, RESULTS.successful, {}

    def transpose(self, state, options):
        (q, r), transpose = state
        transpose_state = (q, r), eqxi.Static(not transpose.value)
        transpose_options = {}
        return transpose_state, transpose_options

    def will_materialise(self):
        return True
