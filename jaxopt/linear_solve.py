import abc
from typing import Dict, TypeVar

import equinox as eqx
from jaxtyping import PyTree, Array

from .linear_operator import AbstractLinearOperator, MatrixLinearOperator, IdentityLinearOperator
from .results import RESULTS
from .str2jax import str2jax


_SolverState = TypeVar("_SolverState")


class AbstractLinearSolver(eqx.Module):
  @abc.abstractmethod
  def init(self, matrix: AbstractLinearOperator) -> _SolverState:
    ...

  @abc.abstractmethod
  def compute(self, state: _SolverState, vector: Array["b"]) -> Tuple[Array["a"], RESULTS, Dict[str, Any]]:
    ...


_cg_token = str2jax("CG")
_cholesky_token = str2jax("Cholesky")
_svd_token = str2jax("SVD")
_lu_token = str2jax("LU")
_qr_token = str2jax("QR")


# Ugly delayed imports because we have the dependency chain
# linear_solve -> AutoLinearSolver -> {CholeskyLinearSolver,...} -> AbstractLinearSolver
# but we want linear_solver and AbstractLinearSolver in the same file.
class AutoLinearSolver(AbstractLinearSolver):
  symmetric: bool = False
  maybe_singular: bool = False

  def init(self, matrix):
    from . import solvers
    if matrix.in_size() == matrix.out_size():
      if self.symmetric:
        if self.maybe_singular:
          return _cg_token, solvers.CGLinearSolver().init(matrix)
        else:
          return _cholesky_token, solvers.CholeskyLinearSolver().init(matrix)
      else:
        if self.maybe_singular:
          return _svd_token, solvers.SVDLinearSolver().init(matrix)
        else:
          return _lu_token, solvers.LULinearSolver().init(matrix)
    else:
      if self.symmetric:
        raise ValueError("Cannot have symmetric non-square matrix")
      return _qr_token, solvers.QRLinearSolver().init(matrix)

  def compute(self, state, vector):
    from . import solvers
    which, state = state
    if which is _cg_token:
      out = solvers.CGLinearSolver().compute(state, vector)
    elif which is _cholesky_token:
      out = solvers.CholeskyLinearSolver().compute(state, vector)
    elif which is _svd_token:
      out = solvers.SVDLinearSolver().compute(state, vector)
    elif which is _lu_token:
      out = solvers.LULinearSolver().compute(state, vector)
    elif which is _qr_token:
      out = solvers.QRLinearSolver().compute(state, vector)
    else:
      assert False
    return out


class LinearSolution(eqx.Module):
  solution: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


# TODO(kidger): gmres, bicg, svd, qr, triangular solvers, diagonal solvers
# TODO(kidger): preconditioners
# TODO(kidger): adjoint
# TODO(kidger): PyTree-valued matrix/vector
def linear_solve(
    matrix: Union[Array["a b"], AbstractLinearOperator, _SolverState],
    vector: Array["b"],
    solver: AbstractLinearSolver = AutoLinearSolver(),
    *,
    is_state: bool = False,
    throw: bool = True
) -> LinearSolveSolution:
  if isinstance(matrix, IdentityLinearOperator):
    return vector
  if is_state:
    state = matrix
  else:
    if isinstance(matrix, (np.ndarray, jnp.ndarray)):
      matrix = MatrixLinearOperator(matrix)
    state = solver.init(matrix)
  del matrix
  solution, result, stats = solver.compute(state, vector)
  has_nans = jnp.any(jnp.isnan(solution))
  result = jnp.where((result == RESULTS.successful) & has_nans, RESULTS.singular, result)
  error_index = unvmap_max(result)
  branched_error_if(
    throw & (results != RESULTS.successful),
    error_index,
    RESULTS.reverse_lookup
  )
  return LinearSolution(solution=solution, result=result, state=state, stats=stats)
