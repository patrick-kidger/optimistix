# TODO(kidger): support non-square matrices
# TODO(kidger): support singular matrices
# TODO(kidger): once the above are done, replace SVD with QR in
#   AutoLinearSolver.
class QR(AbstractLinearSolver):
  def init(self, operator):
    if operator.in_size() != operator.out_size():
      raise ValueError("`QR` may only be used for linear solves with square matrices")
    return jnp.linalg.qr(operator.as_matrix(), mode="reduced")

  def compute(self, state, vector):
    vector, unflatten = jfu.ravel_pytree(vector)
    q, r = state
    solution = unflatten(jsp.linalg.solve_triangular(r, q.T @ vector, lower=False))
    return solution, RESULTS.successful, {}
