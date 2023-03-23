from typing import Optional

import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..linear_solve import AbstractLinearSolver
from ..misc import resolve_rcond
from ..solution import RESULTS
from .misc import (
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


class SVD(AbstractLinearSolver):
    """SVD solver for linear systems.

    This solver can handle any operator, even nonsquare or singular ones. In these
    cases it will return the pseudoinverse solution to the linear system.

    Equivalent to `scipy.linalg.lstsq`.
    """

    rcond: Optional[float] = None

    def init(self, operator, options):
        del options
        svd = jsp.linalg.svd(operator.as_matrix(), full_matrices=False)
        packed_structures = pack_structures(operator)
        return svd, packed_structures

    def compute(self, state, vector, options):
        del options
        (u, s, vt), packed_structures = state
        vector = ravel_vector(vector, packed_structures)
        m, _ = u.shape
        _, n = vt.shape
        rcond = resolve_rcond(self.rcond, n, m, s.dtype)
        mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]
        rank = mask.sum()
        safe_s = jnp.where(mask, s, 1)
        s_inv = jnp.where(mask, 1 / safe_s, 0)
        uTb = jnp.matmul(u.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {"rank": rank}

    def transpose(self, state, options):
        del options
        (u, s, vt), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (vt.T, s, u.T), transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def allow_dependent_columns(self, operator):
        return True

    def allow_dependent_rows(self, operator):
        return True


SVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator. (I.e.
    `N` is the output size and `M` is the input size.)
"""
