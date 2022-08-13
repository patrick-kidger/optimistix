import equinox as eqx


def str2jax(msg: str):
  class M(eqx.Module):
    def __repr__(self):
      return msg
  return M()
