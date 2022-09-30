_SolverState = TypeVar("_SolverState")


class LeastSquaresProblem(AbstractIterativeProblem):
  fn: Callable


class AbstractLeastSquaresSolver(AbstractIterativeSolver):
  pass


class LeastSquaresSolution(eqx.Module):
  optimum: Array
  result: RESULTS
  state: _SolverState
  stats: Dict[str, Array]


def _residual(optimum, _, inputs, __):
  residual_prob, args = inputs
  del inputs

  def objective(_optimum):
    return jnp.sum(residual_prob.fn(_optimum, args)**2)
  return jax.grad(objective)(optimum)


class _ToRootFn(eqx.Module):
  residual_fn: Callable

  def __call__(self, y, args):
    def objective(_y):
      return jnp.sum(self.residual_fn(_y, args)**2)
    return jax.grad(objective)(y)


class _ToMinimiseFn(eqx.Module):
  residual_fn: Callable

  def __call__(self, y, args):
    return jnp.sum(self.residual_fn(y, args)**2)


@eqx.filter_jit
def least_squares_solve(
    residual_fn: Union[Callable, LeastSquaresProblem],
    solver: Union[AbstractLeastSquaresSolver, AbstractMinimiseSolver, AbstractRootFindSolver],
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
    root_prob = RootFindProblem(root_fn, pattern=Pattern(symmetric=True))
    sol = root_find_solve(root_prob, solver, y0, args, options, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return LeastSquaresSolution(optimum=sol.root, result=sol.result, state=sol.state, stats=sol.stats)

  elif isinstance(solver, AbstractMinimiseSolver):
    minimise_fn = _ToMinimiseFn(residual_prob.fn)
    minimise_prob = MinimiseProblem(minimise_fn)
    sol = minimise(minimise_prob, solver, y0, args, options, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return LeastSquaresSolution(optimum=sol.optimum, result=sol.result, state=sol.state, stats=sol.stats)

  else:
    optimum, result, state, stats = iterative_solve(residual_prob, solver, y0, args, options, rewrite_fn=_residual, max_steps=max_steps, adjoint=adjoint, throw=throw)
    return LeastSquaresSolution(optimum=optimum, result=result, state=state, stats=stats)
