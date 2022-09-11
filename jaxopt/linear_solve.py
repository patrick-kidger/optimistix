import abc
from typing import Dict, TypeVar

import equinox as eqx
from jaxtyping import PyTree, Array

from .linear_operator import AbstractLinearOperator, PyTreeLinearOperator, IdentityLinearOperator
from .results import RESULTS
from .str2jax import str2jax


_SolverState = TypeVar("_SolverState")


class AbstractLinearSolver(eqx.Module):
  @abc.abstractmethod
  def init(self, operator: AbstractLinearOperator) -> _SolverState:
    ...

  @abc.abstractmethod
  def compute(self, state: _SolverState, vector: Array["b"]) -> Tuple[Array["a"], RESULTS, Dict[str, Any]]:
    ...


_cg_token = str2jax("CG")
_cholesky_token = str2jax("Cholesky")
_lu_token = str2jax("LU")
_svd_token = str2jax("SVD")


# Ugly delayed imports because we have the dependency chain
# linear_solve -> AutoLinearSolver -> {Cholesky,...} -> AbstractLinearSolver
# but we want linear_solver and AbstractLinearSolver in the same file.
class AutoLinearSolver(AbstractLinearSolver):
  symmetric: bool = False
  maybe_singular: bool = False

  def init(self, operator):
    from . import solvers
    if operator.in_size() == operator.out_size():
      if self.symmetric:
        if self.maybe_singular:
          # CG converges to pseudoinverse/least-squares solution.
          token = _cg_token
          solver = solvers.CG()
        else:
          # Cholesky is pretty much optimal for symmetric dense matrices.
          token = _cholesky_token
          solver = solvers.Cholesky()
      else:
        if self.maybe_singular:
          # SVD converges to pseudoinverse/least-squares solution, and can handle
          # singular matrices.
          # TODO(kidger): make this QR once that solver can handle singular matrices.
          token = _svd_token
          solver = solvers.SVD()
        else:
          # LU is pretty cheap, doesn't require symmetry, doesn't allow
          # singular matrices.
          token = _lu_token
          solver = solvers.LU()
    else:
      if self.symmetric:
        raise ValueError("Cannot have symmetric non-square operator")
      # SVD converges to pseudoinverse/least-squares solution, and can handle
      # rectangular matrices.
      # TODO(kidger): make this QR once that solver can handle nonsquare matrices.
      token = _svd_token
      solver = solvers.SVD()
    return token, solver.init(operator)

  def compute(self, state, vector):
    from . import solvers
    which, state = state
    if which is _cg_token:
      solver = solvers.CG()
    elif which is _cholesky_token:
      solver = solvers.Cholesky()
    elif which is _lu_token:
      solver = solvers.LU()
    elif which is _svd_token:
      solver = solvers.SVD()
    else:
      assert False
    return solver.compute(state, vector)


class LinearSolution(eqx.Module):
  solution: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


_sentinel = object()


# TODO(kidger): gmres, bicg, triangular solvers, diagonal solvers
# TODO(kidger): preconditioners
# TODO(kidger): adjoint
def linear_solve(
    operator: Union[PyTree[Array], AbstractLinearOperator, _SolverState],
    vector: PyTree[Array],
    solver: AbstractLinearSolver = AutoLinearSolver(),
    *,
    state: PyTree[Array] = _sentinel,
    throw: bool = True
) -> LinearSolveSolution:
  if state is _sentinel:
    vector_structure = jtu.tree_map(jnp.shape, vector)
    if isinstance(operator, AbstractLinearOperator):
      if vector != operator.in_structure():
        raise ValueError("Vector and operator structures do not match")
      if isinstance(operator, IdentityLinearOperator):
        return vector
    else:
      operator = PyTreeLinearOperator(operator, vector_structure)
    state = solver.init(operator)
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
