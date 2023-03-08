from typing import Optional

import equinox.internal as eqxi
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu

from ..linear_solve import AbstractLinearSolver
from ..misc import ordered_ravel, ordered_unravel, resolve_rcond
from ..solution import RESULTS


class SVD(AbstractLinearSolver):
    """SVD solver for linear systems.

    This solver can handle singular operators.

    This solver is slightly more expensive than [`optimistix.QR`][], but is slightly
    more numerically stable.

    Equivalent to `scipy.linalg.lstsq`.
    """

    rcond: Optional[float] = None

    def init(self, operator, options):
        del options
        structures = operator.out_structure(), operator.in_structure()
        leaves, treedef = jtu.tree_flatten(structures)  # handle nonhashable pytrees
        structures = eqxi.Static((leaves, treedef))
        svd = jsp.linalg.svd(operator.as_matrix(), full_matrices=False)
        return svd, structures

    def compute(self, state, vector, options):
        del options
        (u, s, vt), structures = state
        leaves, treedef = structures.value
        out_structure, in_structure = jtu.tree_unflatten(treedef, leaves)
        vector = ordered_ravel(vector, out_structure)
        m, _ = u.shape
        _, n = vt.shape
        rcond = resolve_rcond(self.rcond, n, m, s.dtype)
        mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]
        rank = mask.sum()
        safe_s = jnp.where(mask, s, 1)
        s_inv = jnp.where(mask, 1 / safe_s, 0)
        uTb = jnp.matmul(u.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
        solution = ordered_unravel(solution, in_structure)
        return solution, RESULTS.successful, {"rank": rank}

    def pseudoinverse(self, operator):
        return True

    def transpose(self, state, options):
        del options
        (u, s, vt), structures = state
        leaves, treedef = structures.value
        out_structure, in_structure = jtu.tree_unflatten(treedef, leaves)
        leaves, treedef = jtu.tree_flatten((in_structure, out_structure))
        transpose_structures = eqxi.Static((leaves, treedef))
        transpose_state = (vt.T, s, u.T), transpose_structures
        transpose_options = {}
        return transpose_state, transpose_options


SVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator. (I.e.
    `N` is the output size and `M` is the input size.)
"""
