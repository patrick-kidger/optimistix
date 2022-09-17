_SolverState = TypeVar("_SolverState")


class AbstractLeastSquaresSolver(AbstractIterativeSolver):
  pass


class _ToRootFn(eqx.Module):
  residual_fn: Callable

  def __call__(self, y, args):
    def objective(_y):
      return jnp.sum(self.residual_fn(_y, args)**2)
    return jax.grad(objective)(y)


class LeastSquaresProblem(AbstractIterativeProblem):
  fn: Callable


class LeastSquaresSolution(eqx.Module):
  optimum: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def _residual(root, _, inputs, __):
  residual_fn, args = inputs
  del inputs

  def objective(_y):
    return jnp.sum(residual_fn(_y, args)**2)
  return jax.grad(objective)(root)


def least_squares_solve(
    residual_fn: Union[Callable, LeastSquaresProblem],
    solver: Union[AbstractLeastSquaresSolver, AbstractRootFindSolver],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 16,
    adjoint: AbstractAdjoint = ImplicitAdjoint()
    throw: bool = True,
):
  if isinstance(residual_fn, LeastSquaresProblem):
    residual_prob = residual_fn
  else:
    residual_prob = LeastSquaresProblem(residual_fn)
  del residual_fn

  if isinstance(solver, AbstractRootFindSolver):
    root_fn = _ToRootFn(residual_prob.fn)
    root_prob = RootFindProblem(root_fn, symmetric=True)
    sol = root_find_solve(root_prob, solver, y0, args, options, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return LeastSquaresSolution(optimum=sol.root, result=sol.result, state=sol.state, stats=sol.stats)
  else:
    optimum, result, state, stats = iterative_solve(residual_prob, solver, y0, args, options, rewrite_fn=_residual, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return LeastSquaresSolution(optimum=optimum, result=result, state=state, stats=stats)
