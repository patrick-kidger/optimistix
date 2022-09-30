def resolve_rcond(rcond, n, m, dtype):
  if rcond is None:
    return jnp.finfo(dtype).eps * max(n, m)
  else:
    return jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)
