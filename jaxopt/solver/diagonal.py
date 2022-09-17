class Diagonal(AbstractLinearSolver):
  def init(self, operator, options):
    del options
    if operator.in_size() != operator.out_size():
      raise ValueError("`Diagonal` may only be used for linear solves with square matrices")
    if "diagonal" not in operator.patterns:
      raise ValueError("`Diagonal` may only be used for linear solves with diagonal matrices")
    return operator

  def compute(self, state, vector, options):
    operator = state
    del state, options
    if "unit_diagonal" in operator.patterns:
      solution = vector
    else:
      # TODO(kidger): do diagonal solves more efficiently than this.
      vector, unflatten = jfu.ravel_pytree(vector)
      solution = unflatten(vector / jnp.diag(operator.as_matrix()))
    return solution, RESULTS.successful, {}
