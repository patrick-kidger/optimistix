_SolverState = TypeVar("_SolverState")


class MinimiseProblem(AbstractIterativeProblem):
  fn: Callable


class AbstractMinimiseSolver(AbstractIterativeSolver):
  pass


class MinimiseSolution(eqx.Module):
  optimum: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def _minimum(optimum, _, inputs, __):
  minimise_prob, args = inputs
  del inputs
  return jax.grad(minimise_prob.fn)(optimum, args)


class _ToRootFn(eqx.Module):
  minimise_fn: Callable

  def __call__(self, y, args):
    return jax.grad(self.minimise_fn)(y, args)


@eqx.filter_jit
def minimise_solve(
    minimise_fn: Union[Callable, MinimiseProblem],
    solver: Union[AbstractMinimiseSolver, AbstractRootFindSolver],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 16,
    adjoint: AbstractAdjoint = ImplicitAdjoint()
    throw: bool = True,
):
  if isinstance(residual_fn, MinimiseProblem):
    minimise_prob = minimise_fn
  else:
    minimise_prob = MinimiseProblem(minimise_fn)
  del minimise_fn

  if isinstance(solver, AbstractMinimiseSolver):
    root_fn = _ToRootFn(minimise_prob.fn)
    root_prob = RootFindProblem(root_fn, pattern=Pattern(symmetric=True))
    sol = root_find_solve(root_prob, solver, y0, args, options, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return MinimiseSolution(optimum=sol.root, result=sol.result, state=sol.state, stats=sol.stats)
  else:
    optimum, result, state, stats = iterative_solve(residual_prob, solver, y0, args, options, rewrite_fn=_minimum, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return MinimiseSolution(optimum=optimum, result=result, state=state, stats=stats)
