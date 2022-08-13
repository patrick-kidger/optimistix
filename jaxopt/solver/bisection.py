import equinox as eqx
from diffrax.misc import rms_norm, error_if

from ..results import RESULTS
from ..root_finding import AbstractRootFindSolver


class _BisectionState(eqx.Module):
  lower: Scalar
  upper: Scalar
  sign: Scalar
  error: Scalar


class Bisection(AbstractRootFindSolver):
  tol: float

  def init(self, root_fun, y: Scalar, args, options):
    upper = options["upper"]
    lower = options["lower"]
    if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
      raise ValueError("Bisection can only be used to find the roots of a function taking a scalar input")
    if jnp.shape(root_fun(y)) != ():
      raise ValueError("Bisection can only be used to find the roots of a function producing a scalar output")
    upper_sign = jnp.sign(upper)
    lower_sign = jnp.sign(lower)
    error_if(upper_sign == lower_sign, "The root is not contained in [lower, upper]")
    return _BisectionState(lower=lower, upper=upper, sign=upper_sign, error=jnp.asarray(jnp.inf))

  def step(self, root_fun, y: Scalar, args, state):
    del y
    new_y = state.lower + 0.5 * (state.upper - state.lower)
    error = root_fun(new_y, args)
    too_large = state.sign * error
    new_lower = jnp.where(too_large, new_y, state.lower)
    new_upper = jnp.where(too_large, state.uppwer, new_y)
    return new_y, _BisectionState(lower=new_lower, upper=new_upper, sign=state.sign, error=error)

  def terminate(self, root_fun, y: Scalar, args, state):
    del root_fun, y, args
    return state.error < self.tol, RESULTS.successful
