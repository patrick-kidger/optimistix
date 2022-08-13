class CG(AbstractLinearSolver):
  rtol: float
  atol: float
  norm: Callable = rms_norm
  materialise: bool = False
  max_steps: Optional[int] = None

  def init(self, matrix):
    if matrix.in_size() != matrix.out_size():
      raise ValueError("`CG` may only be used for linear solves with square matrices")
    if self.materialise:
      matrix = matrix.materialise()
    return matrix

  # This differs from jax.scipy.sparse.linalg.cg in:
  # 1. We use a slightly different termination condition -- rtol and atol are used in
  #    a conventional way, and `scale` is vector-valued and recomputed on every step.
  #    (Instead of being scalar-valued and computed just at the start.)
  # 2. We return the number of steps, and whether or not the solve succeeded, as
  #    additional information.
  # 3. We don't try to support complex numbers.
  def compute(self, state, vector):
    preconditioner = IdentityLinearOperator(matrix.in_size()) # TODO(kidger): offer this as a proper argument
    y0 = jnp.zeros(matrix.in_size(), vector.dtype)  # TODO(kidger): offer this as a proper argument
    r0 = vector - state.mv(y0)
    p0 = preconditioner.mv(r0)
    gamma0 = jnp.sum(r0 * p0)
    initial_value = (y0, r0, p0, gamma0, 0)

    def cond_fun(value):
      _, r, _, _, step = value
      scale = self.atol + self.rtol * vector
      out = self.norm(r / scale) > 1
      if self.max_steps is not None:
        out = out & (step < self.max_steps)
      return out

    def body_fun(value):
      y, r, p, gamma, step = value
      mat_p = state.mv(p)
      alpha = gamma / jnp.sum(p * mat_p)
      y = y + alpha * p
      r = r - alpha * mat_p
      z = preconditioner.mv(r)
      gamma_prev = gamma
      gamma = jnp.sum(r * z)
      beta = gamma / gamma_prev
      p = z + beta * p
      return y, r, p, gamma, step + 1

    solution, _, _, _, num_steps = lax.while_loop(cond_fun, body_fun, initial_value)
    if self.max_steps is None:
      result = RESULTS.successful
    else:
      result = jnp.where(num_steps == self.max_steps, RESULTS.max_steps_reached, RESULTS.successful)
    return solution, result, {"num_steps": num_steps, "max_steps": self.max_steps}
