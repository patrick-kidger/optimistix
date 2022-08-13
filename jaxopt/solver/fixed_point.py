import equinox as eqx
from diffrax.misc import rms_norm, ω
from jaxtyping import Array

from ..fixed_point import AbstractFixedPointSolver
from ..results import RESULTS


class FixedPointIteration(AbstractFixedPointSolver):
  tol: float
  norm: Callable = rms_norm

  def init(self, fixed_point_fun, y, args, options: None) -> Array[""]:
    del fixed_point_fun, y, args, options
    return jnp.array(jnp.inf)

  def step(self, fixed_point_fun, y, args, state: Array[""]):
    y_next = fixed_point_fun(y, args)
    return y_next, self.norm((y**ω - y_next**ω).ω)

  def terminate(self, fixed_point_fun, y, args, state: Array[""]):
    error = state
    return error < self.tol, RESULTS.successful
