import abc
from typing import Dict, TypeVar, Union

import equinox as eqx
from diffrax.misc import bounded_while_loop
from jaxtyping import PyTree, Array

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .results import RESULTS
from .root_find import AbstractRootFindSolver


_SolverState = TypeVar("_SolverState")


class AbstractFixedPointSolver(eqx.Module):
  @abc.abstractmethod
  def init(self, fixed_point_fun: Callable, y: PyTree[Array], args: PyTree, options: Dict[str, Any]) -> _SolverState:
    ...

  @abc.abstractmethod
  def step(self, fixed_point_fun: Callable, y: PyTree[Array], args: PyTree, state: _SolverState) -> Tuple[PyTree[Array], _SolverState]:
    ...

  @abc.abstractmethod
  def terminate(self, fixed_point_fun: Callable, y: PyTree[Array], args: PyTree, state: _SolverState) -> bool:
    ...


class _RootFindToFixedPoint(AbstractFixedPointSolver):
  solver: AbstractRootFindSolver

  def init(self, fixed_point_fun, y, args, options):
    def root_fn(y, args):
      return fixed_point_fun(y, args) - y
    return solver.init(root_fn, y, args, options)

  def step(self, fixed_point_fun, y, args, state):
    def root_fn(y, args):
      return fixed_point_fun(y, args) - y
    return solver.step(root_fn, y, args, state)

  def terminate(self, fixed_point_fun, y, args, state):
    def root_fn(y, args):
      return fixed_point_fun(y, args) - y
    return solver.terminate(root_fn, y, args, state)


class FixedPointSolution(eqx.Module):
  fixed_point: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def fixed_point_solve(
    fixed_point_fun: Callable,
    solver: Union[AbstractFixedPointSolver, AbstractRootFindSolver],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 16,
    adjoint: AbstractAdjoint = ImplicitAdjoint()  # TODO
    throw: bool = True,
):

  assert jax.eval_shape(lambda: y0)() == jax.eval_shape(fixed_point_fun, y0, args)()

  if isinstance(solver, AbstractRootFindSolver):
    solver = _RootFindingToFixedPoint(solver)

  init_state = solver.init(root_fun, y0, args, options)
  init_val = (0, y0, init_state)

  def cond_fun(carry):
    _, y, state = carry
    terminate, _  = solver.terminate(y, args, state)
    return jnp.invert(terminate)

  def body_fun(carry, _):
    num_steps, y, state = carry
    new_y, new_state = solver.step(y, args, state)
    return num_steps + 1, new_y, new_state

  final_val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps, base=4)
  num_steps, final_y, final_state = final_val
  terminate, result = solver.terminate(final_y, args, final_state)
  result = jnp.where(result == RESULTS.successful, jnp.where(terminate, RESULTS.successful, RESULTS.max_steps_reached), result)
  error_index = unvmap_max(result)
  branched_error_if(
    throw & (results != RESULTS.successful),
    error_index,
    RESULTS.reverse_lookup
  )
  stats = {"num_steps": num_steps, "max_steps": max_steps}
  return FixedPointSolution(fixed_point=final_y, result=result, state=final_state, stats=stats)

