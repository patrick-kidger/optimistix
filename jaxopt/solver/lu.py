class LU(AbstractLinearSolver):
  def init(self, matrix):
    if matrix.in_size() != matrix.out_size():
      raise ValueError("`LU` may only be used for linear solves with square matrices")
    return jsp.linalg.lu_factor(matrix.as_matrix())

  def compute(self, state, vector):
    solution = jsp.linalg.lu_solve(state, vector)
    return solution, RESULTS.successful, {}
