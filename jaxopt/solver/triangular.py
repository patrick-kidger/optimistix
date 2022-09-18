class Triangular(AbstractLinearSolver):
  def init(self, operator, options):
    del options
    if operator.in_size() != operator.out_size():
      raise ValueError("`Triangular` may only be used for linear solves with square matrices")
    if "triangular" not in operator.pattern:
      raise ValueError("`Triangular` may only be used for linear solves with triangular matrices")
    return (operator.as_matrix(), Static("lower_triangular" in operator.pattern), Static("unit_diagonal" in operator.pattern))

  def compute(self, state, vector, options):
    matrix, lower, unit_diagonal = state
    del state, options
    vector, unflatten = jfu.ravel_pytree(vector)
    solution = unflatten(jsp.linalg.solve_triangular(matrix, vector, lower=lower.value, unit_diagonal=unit_diagonal.value))
    return solution, RESULTS.successful, {}
