class SVD(AbstractLinearSolver):
  rcond: Optional[float] = None

  def init(self, operator):
    return jsp.linalg.svd(operator.as_matrix(), full_matrices=False)

  def compute(self, state, vector):
    vector, unflatten = jfu.ravel_pytree(vector)
    u, s, vt = state
    m, _ = u.shape
    _, n = vt.shape
    dtype = vector.dtype
    if self.rcond is None:
      rcond = jnp.finfo(dtype).eps * max(n, m)
    else:
      rcond = jnp.where(self.rcond < 0, jnp.finfo(dtype).eps, self.rcond)
    mask = s >= jnp.array(rcond, dtype=s.dtype) * s[0]
    rank = mask.sum()
    safe_s = jnp.where(mask, s, 1).astype(a.dtype)
    s_inv = jnp.where(mask, 1 / safe_s, 0)
    uTb = jnp.matmul(u.conj().T, b, precision=lax.Precision.HIGHEST)
    solution = unflatten(jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST))
    return solution, RESULTS.successful, {"rank": rank}
