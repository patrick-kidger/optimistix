from typing import Optional

import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float

from ..linear_solve import AbstractLinearSolver
from ..linear_operator import is_lower_triangular, is_upper_triangular, has_unit_diagonal
from ..misc import resolve_rcond
from ..solution import RESULTS


class Triangular(AbstractLinearSolver):
    rcond: Optional[float] = None

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
        )

    def compute(self, state, vector, options):
        matrix, lower, unit_diagonal = state
        del state, options
        vector, unflatten = jfu.ravel_pytree(vector)
        solution = solve_triangular(matrix, vector, lower, unit_diagonal, self.rcond)
        solution = unflatten(solution)
        return solution, RESULTS.successful, {}

    def pseudoinverse(self, operator):
        return True

    def transpose(self, state, options):
        matrix, lower, unit_diagonal = state
        transpose_state = matrix.T, not lower.value, unit_diagonal
        transpose_options = {}
        return transpose_state, transpose_options


def solve_triangular(
    matrix: Float[Array, "n n"],
    vector: Float[Array, " n"],
    lower: bool,
    unit_diagonal: bool,
    rcond: Optional[float],
) -> Float[Array, " n"]:
    # This differs from jax.scipy.linalg.solve_triangular in that it will return
    # pseudoinverse solutions if the matrix is singular.

    n, m = matrix.shape
    (k,) = vector.shape
    assert n == m
    assert n == k
    if unit_diagonal:
        # Unit diagonal implies nonsingular, so use the fact that this lowers to an XLA
        # primitive for efficiency.
        return jsp.linalg.solve_triangular(
            matrix, vector, lower=lower, unit_diagonal=unit_diagonal
        )
    rcond = resolve_rcond(rcond, n, m, matrix.dtype)
    cutoff = rcond * jnp.max(jnp.diag(matrix))

    def scan_fn(_solution, _input):
        _i, _row = _input
        _val = jnp.dot(_solution, _row, precision=lax.Precision.HIGHEST)
        _row_i = _row[_i]
        _row_i = jnp.where(jnp.abs(_row_i) >= cutoff, _row_i, jnp.inf)
        _solution = _solution.at[_i].set((vector[_i] - _val) / _row_i)
        return _solution, None

    init_solution = jnp.zeros_like(vector)
    solution, _ = lax.scan(
        scan_fn, init_solution, (jnp.arange(n), matrix), reverse=not lower
    )
    return solution
