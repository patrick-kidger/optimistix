import equinox as eqx
from diffrax.misc import rms_norm, ω

from ..fixed_point import AbstractFixedPointSolver
from ..results import RESULTS


class FixedPointIteration(AbstractFixedPointSolver):
  tol: float
  norm: Callable = rms_norm

  def init(self, fixed_point_fn, y, args, options) -> Scalar:
    del fixed_point_fn, y, args, options
    return jnp.array(jnp.inf)

  def step(self, fixed_point_fn, y, args, options, state: Scalar):
    y_next = fixed_point_fn(y, args)
    return y_next, self.norm((y**ω - y_next**ω).ω)

  def terminate(self, fixed_point_fn, y, args, options, state: Scalar):
    error = state
    return error < self.tol, RESULTS.successful
