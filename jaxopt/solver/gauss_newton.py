def _small(diffsize: Scalar) -> bool:
  # TODO(kidger): make a more careful choice here -- the existence of this
  # function is pretty ad-hoc.
  resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
  return diffsize < resolution


def _diverged(rate: Scalar) -> bool:
  return ~jnp.isfinite(rate) | (rate > 2)


def _converged(factor: Scalar, tol: Scalar) -> bool:
  return (factor > 0) & (factor < tol)


class _GaussNewtonState(eqx.Module):
  step: Scalar
  diffsize: Scalar
  diffsize_prev: Scalar


class GaussNewton(AbstractLeastSquaresSolver):
  rtol: float
  atol: float
  kappa: float = 1e-2
  norm: Callable = rms_norm
  linear_solver: AbstractLinearSolver = AutoLinearSolver()

  def init(self, residual_prob, y, args, options):
    del residual_prob, y, args, options
    return _GaussNewtonState(step=jnp.array(0), diffsize=jnp.array(0.0), diffsize_prev=jnp.array(0.0))

  def step(self, residual_prob, y, args, options, state):
    del options
    residuals = residual_prob.fn(y, args)
    jac = JacobianLinearOperator(residual_prob.fn, y, args)
    diff = linear_solve(jac, residuals, self.linear_solver).solution
    scale = (self.atol + self.rtol * y**ω).ω
    diffsize = self.norm((diff**ω / scale**ω).ω)
    new_y = (y**ω - diff**ω).ω
    new_state = _GaussNewtonState(step=state.step + 1, diffsize=diffsize, diffsize_prev=state.diffsize)
    return new_y, new_state

  def terminate(self, residual_prob, y, args, options, state):
    del residual_prob, y, args, options
    at_least_two = state.step >= 2
    rate = state.diffsize / state.diffsize_prev
    factor = state.diffsize * rate / (1  - rate)
    small = _small(state.diffsize)
    diverged = _diverged(rate)
    converged = _converged(factor, self.kappa)
    terminate = at_least_two & (small | diverged | converged)
    result = jnp.where(converged, RESULTS.successful, RESULTS.nonconvergence)
    result = jnp.where(diverged, RESULTS.divergence, result)
    result = jnp.where(small, RESULTS.successful, result)
    return terminate, result
