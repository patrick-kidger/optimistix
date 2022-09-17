_SolverState = TypeVar("_SolverState")


class RootFindProblem(AbstractIterativeProblem):
  fn: Callable
  patterns: Patterns = Patterns()


class AbstractRootFindSolver(AbstractIterativeSolver):
  pass


class RootFindSolution(eqx.Module):
  root: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def _root(root, _, inputs, __):
  root_fn, args = inputs
  del inputs
  return root_fn(root, args)


def root_find_solve(
    root_fn: Union[Callable, RootFindProblem],
    solver: AbstractRootFindSolver,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 16,
    adjoint: AbstractAdjoint = ImplicitAdjoint()
    throw: bool = True,
):
  if isinstance(root_fn, RootFindProblem):
    root_prob = root_fn
  else:
    root_prob = RootFindProblem(root_prob)
  del root_fn

  root, result, state, stats = iterative_solve(root_prob, solver, y0, args, options, rewrite_fn=_root, patterns=root_prob.patterns, max_steps=max_steps, adjoint=adjoint, throw=throw)
  return RootFindSolution(root=root, result=result, state=state, stats=stats)

