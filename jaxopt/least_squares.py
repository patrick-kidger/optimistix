_SolverState = TypeVar("_SolverState")


class AbstractLeastSquaresSolver(AbstractIterativeSolver):
  pass


class _ToRootFn(eqx.Module):
  residual_fn: Callable

  def __call__(self, y, args):
    return jax.jacfwd(self.residual_fn)(y, args)


class LeastSquaresSolution(eqx.Module):
  optimum: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def _residual(root, _, inputs, __):
  residual_fn, _, args = inputs
  del inputs
  return jax.jacfwd(residual_fn)(root, args)


def least_squares_solve(
    residual_fn: Callable,
    solver: Union[AbstractLeastSquaresSolver, AbstractRootFindSolver],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 16,
    adjoint: AbstractAdjoint = ImplicitAdjoint()
    throw: bool = True,
):
  if isinstance(solver, AbstractRootFindSolver):
    root_fn = _ToRootFn(residual_fn)
    sol = root_find_solve(root_fn, solver, y0, args, options, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return LeastSquaresSolution(optimum=sol.root, result=sol.result, state=sol.state, stats=sol.stats)
  else:
    optimum, result, state, stats = iterative_solve(residual_fn, solver, y0, args, options, rewrite_fn=_residual, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return LeastSquaresSolution(optimum=optimum, result=result, state=state, stats=stats)
