class Cholesky(AbstractLinearSolver):
  def init(self, matrix):
    if matrix.in_size() != matrix.out_size():
      raise ValueError("`Cholesky` may only be used for linear solves with square matrices")
    # Fix lower triangular, so that the boolean flag doesn't get needlessly promoted
    # to a tracer anywhere.
    factor, lower = jsp.linalg.cho_factor(matrix.as_matrix())
    assert lower is False
    return factor

  def solve(self, state, vector):
    return jsp.linalg.cho_solve((state, False), vector)
