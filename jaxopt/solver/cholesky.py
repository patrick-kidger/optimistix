class Cholesky(AbstractLinearSolver):
  normal: bool = False
  jitter: float = 0

  def init(self, operator, options):
    del options
    if self.normal:
      matrix = operator.as_matrix()
      _, n = matrix.shape
      factor, lower = jsp.linalg.cho_factor(matrix.T @ matrix + self.jitter * jnp.eye(n))
      state = matrix, factor
    else:
      if operator.in_size() != operator.out_size():
        raise ValueError("`Cholesky(..., normal=False)` may only be used for linear solves with square matrices")
      if not operator.patterns.symmetric:
        raise ValueError("`Cholesky(..., normal=False)` may only be used for symmetric linear operators")
      state, lower = jsp.linalg.cho_factor(operator.as_matrix())
    # Fix lower triangular, so that the boolean flag doesn't get needlessly promoted
    # to a tracer anywhere.
    assert lower is False
    return state

  def solve(self, state, vector, options):
    del options
    vector, unflatten = jfu.ravel_pytree(vector)
    if self.normal:
      matrix, factor = state
      vector = matrix.T @ vector
    else:
      factor = state
    return unflatten(jsp.linalg.cho_solve((factor, False), vector))
