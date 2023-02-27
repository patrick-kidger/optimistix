from typing import Optional

import equinox.internal as eqxi
import jax.numpy as jnp
import jax.tree_util as jtu

from ..linear_solve import AbstractLinearSolver
from ..misc import ordered_ravel, ordered_unravel
from ..solution import RESULTS
from .triangular import solve_triangular


class QR(AbstractLinearSolver):
    rcond: Optional[float] = None

    def init(self, operator, options):
        del options
        matrix = operator.as_matrix()
        m, n = matrix.shape
        transpose = n > m
        if transpose:
            matrix = matrix.T
        qr = jnp.linalg.qr(matrix, mode="reduced")
        structures = operator.out_structure(), operator.in_structure()
        leaves, treedef = jtu.tree_flatten(structures)  # handle nonhashable pytrees
        structures = eqxi.Static((leaves, treedef))
        return qr, transpose, structures

    def compute(self, state, vector, options):
        (q, r), transpose, structures = state
        del state, options
        leaves, treedef = structures.value
        out_structure, in_structure = jtu.tree_unflatten(treedef, leaves)
        vector = ordered_ravel(vector, out_structure)
        if transpose:
            # Minimal norm solution if ill-posed
            solution = q @ solve_triangular(
                r.T, vector, lower=True, unit_diagonal=False, rcond=self.rcond
            )
        else:
            # Least squares solution if ill-posed
            solution = solve_triangular(
                r, q.T @ vector, lower=False, unit_diagonal=False, rcond=self.rcond
            )
        solution = ordered_unravel(solution, in_structure)
        return solution, RESULTS.successful, {}

    def pseudoinverse(self, operator):
        return True

    def transpose(self, state, options):
        (q, r), transpose, structures = state
        leaves, treedef = structures.value
        out_structure, in_structure = jtu.tree_unflatten(treedef, leaves)
        leaves, treedef = jtu.tree_flatten((in_structure, out_structure))
        transpose_structures = eqxi.Static((leaves, treedef))
        transpose_state = (q, r), not transpose, transpose_structures
        transpose_options = {}
        return transpose_state, transpose_options
