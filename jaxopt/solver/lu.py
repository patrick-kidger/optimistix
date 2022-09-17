class LU(AbstractLinearSolver):
  def init(self, operator, options):
    del options
    if operator.in_size() != operator.out_size():
      raise ValueError("`LU` may only be used for linear solves with square matrices")
    return jsp.linalg.lu_factor(operator.as_matrix())

  def compute(self, state, vector, options):
    del options
    vector, unflatten = jfu.ravel_pytree(vector)
    solution = unflatten(jsp.linalg.lu_solve(state, vector))
    return solution, RESULTS.successful, {}
