import abc
from typing import Dict, TypeVar

import equinox as eqx
from jaxtyping import PyTree, Array

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .results import RESULTS


_SolverState = TypeVar("_SolverState")


class AbstractRootFindSolver(eqx.Module):
  @abc.abstractmethod
  def init(self, root_fun: Callable, y: PyTree[Array], args: PyTree, options: Dict[str, Any]) -> _SolverState:
    ...

  @abc.abstractmethod
  def step(self, root_fun: Callable, y: PyTree[Array], args: PyTree, state: _SolverState) -> Tuple[PyTree[Array], _SolverState]:
    ...

  @abc.abstractmethod
  def terminate(self, root_fun: Callable, y: PyTree[Array], args: PyTree, state: _SolverState) -> bool:
    ...


class RootFindSolution(eqx.Module):
  root: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def root_find_solve(
    root_fun: Callable
    solver: AbstractRootFindSolver,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 16,  # TODO: base, checkpointing?
    adjoint: AbstractAdjoint = ImplicitAdjoint()  # TODO
    throw: bool = True,
):

  init_state = solver.init(root_fun, y0, args, options)
  init_val = (0, y0, init_state)

  def cond_fun(carry):
    _, y, state = carry
    terminate, _ = solver.terminate(y, args, state)
    return jnp.invert(terminate)

  def body_fun(carry, _):
    num_steps, y, state = carry
    new_y, new_state = solver.step(y, args, state)
    return num_steps + 1, new_y, new_state

  final_val = bounded_while_loop(cond_fun, body_fun, init_val, max_steps)
  num_steps, root, final_state = final_val
  terminate, result = solver.terminate(final_y, args, final_state)
  result = jnp.where(result == RESULTS.successful, jnp.where(terminate, RESULTS.successful, RESULTS.max_steps_reached), result)
  error_index = unvmap_max(result)
  branched_error_if(
    throw & (results != RESULTS.successful),
    error_index,
    RESULTS.reverse_lookup
  )
  stats = {"num_steps": num_steps, "max_steps": max_steps}
  return RootFindSolution(root=root, result=result, state=final_state, stats=stats)

