class SVD(AbstractLinearSolver):
  maybe_singular: bool = True
  rcond: Optional[float] = None

  def is_maybe_singular(self):
    return self.maybe_singular

  def init(self, operator, options):
    del options
    return jsp.linalg.svd(operator.as_matrix(), full_matrices=False)

  def compute(self, state, vector, options):
    del options
    vector, unflatten = jfu.ravel_pytree(vector)
    u, s, vt = state
    m, _ = u.shape
    _, n = vt.shape
    dtype = vector.dtype
    rcond = resolve_rcond(self.rcond, n, m, dtype)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]
    rank = mask.sum()
    safe_s = jnp.where(mask, s, 1).astype(a.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)
    uTb = jnp.matmul(u.conj().T, b, precision=lax.Precision.HIGHEST)
    solution = unflatten(jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST))
    return solution, RESULTS.successful, {"rank": rank}

  def transpose(self, state, options):
    transpose_state = vt.T, s, u.T
    transpose_options = {}
    return transpose_state, transpose_options
