class AbstractAdjoint(eqx.Module):
  # warning: not a stable API
  @abc.abstractmethod
  def apply(self, primal_fn: Callable, rewrite_fn: Callable, inputs: PyTree[Array], closure: Any, pattern: Pattern):
    ...

class RecursiveCheckpointAdjoint(AbstractAdjoint):
  def apply(self, primal_fn, rewrite_fn, inputs, closure, pattern):
    del rewrite_fn, pattern
    return primal_fl(inputs, closure, reverse_autodiffable=True)


class ImplicitAdjoint(AbstractAdjoint):
  linear_solver: AbstractLinearSolver = AutoLinearSolver()

  def apply(self, primal_fn, rewrite_fn, inputs, closure, pattern):
    primal_fn = ft.partial(primal_fn, reverse_autodiffable=False)
    return implicit_jvp(primal_fn, rewrite_fn, inputs, closure, pattern, self.linear_solver)
