_SolverState = TypeVar("_SolverState")


class AbstractFixedPointSolver(AbstractIterativeSolver):
  pass


class _RootFindToFixedPoint(AbstractFixedPointSolver):
  solver: AbstractRootFindSolver

  def init(self, fixed_point_fn, y, args, options):
    def root_fn(y, args):
      return fixed_point_fn(y, args) - y
    return solver.init(root_fn, y, args, options)

  def step(self, fixed_point_fn, y, args, state):
    def root_fn(y, args):
      return fixed_point_fn(y, args) - y
    return solver.step(root_fn, y, args, state)

  def terminate(self, fixed_point_fn, y, args, state):
    def root_fn(y, args):
      return fixed_point_fn(y, args) - y
    return solver.terminate(root_fn, y, args, state)


class FixedPointSolution(eqx.Module):
  fixed_point: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def fixed_point_solve(
    fixed_point_fn: Callable,
    solver: Union[AbstractFixedPointSolver, AbstractRootFindSolver],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 16,
    adjoint: AbstractAdjoint = ImplicitAdjoint()
    throw: bool = True,
):
  assert jax.eval_shape(lambda: y0)() == jax.eval_shape(fixed_point_fn, y0, args)()
  if isinstance(solver, AbstractRootFindSolver):
    solver = _RootFindingToFixedPoint(solver)
  fixed_point, result, state, stats = iterative_solve(fixed_point_fn, solver, y0, args, options, rewrite_fn=_fixed_point, max_steps=max_steps, adjoint=adjoint, throw=throw)
  return FixedPointSolution(fixed_point=fixed_point, result=result, state=final_state, stats=stats)

