class AbstractAdjoint(eqx.Module):
  # warning: not a stable API
  @abc.abstractmethod
  def apply(self, primal_fn, rewrite_fn, inputs, closure):
    ...

class RecursiveCheckpointAdjoint(AbstractAdjoint):
  def apply(self, primal_fn, rewrite_fn, inputs, closure):
    del rewrite_fn
    return primal_fl(inputs, closure, reverse_autodiffable=True)


class ImplicitAdjoint(AbstractAdjoint):
  linear_solver: AbstractLinearSolver = AutoLinearSolver()

  def apply(self, primal_fn, rewrite_fn, inputs, closure):
    primal_fn = ft.partial(primal_fn, reverse_autodiffable=False)
    return implicit_jvp(primal_fn, rewrite_fn, inputs, closure, linear_solver)
