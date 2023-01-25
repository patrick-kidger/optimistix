from typing import Optional

import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..linear_solve import AbstractLinearSolver
from ..misc import resolve_rcond
from ..solution import RESULTS


class SVD(AbstractLinearSolver):
    rcond: Optional[float] = None

    def init(self, operator, options):
        del options
        return jsp.linalg.svd(operator.as_matrix(), full_matrices=False)

    def compute(self, state, vector, options):
        del options
        vector, unflatten = jfu.ravel_pytree(vector)
        u, s, vt = state
        m, _ = u.shape
        _, n = vt.shape
        rcond = resolve_rcond(self.rcond, n, m, s.dtype)
        mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]
        rank = mask.sum()
        safe_s = jnp.where(mask, s, 1)
        s_inv = jnp.where(mask, 1 / safe_s, 0)
        uTb = jnp.matmul(u.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = unflatten(
            jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
        )
        return solution, RESULTS.successful, {"rank": rank}

    def pseudoinverse(self, operator):
        return True

    def transpose(self, state, options):
        del options
        u, s, vt = state
        transpose_state = vt.T, s, u.T
        transpose_options = {}
        return transpose_state, transpose_options
