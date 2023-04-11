from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree

from ..custom_types import Scalar, sentinel
from ..linear_operator import AbstractLinearOperator
from ..minimise import AbstractMinimiser
from ..misc import max_norm
from ..solution import RESULTS


def _small(diffsize: Scalar) -> bool:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> bool:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: Scalar) -> bool:
    return (factor > 0) & (factor < tol)


class QNState(eqx.Module):
    step: Scalar
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    # TODO(raderj): can we specifically say that we could encounter a sentinel
    # value? Maybe a custom sentinel type
    vector: Optional[PyTree[ArrayLike]] | object
    operator: Optional[AbstractLinearOperator] | object


class AbstractQuasiNewton(AbstractMinimiser):
    def init(self, problem, y, args, options):
        del problem, y, args, options
        return QNState(
            step=jnp.array(0),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=jnp.array(RESULTS.successful),
            vector=sentinel,
            operator=sentinel,
        )

    def terminate(self, problem, y, args, options, state):
        at_least_two = state.step >= 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.kappa)
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (at_least_two & (small | diverged | converged))
        result = jnp.where(diverged, RESULTS.divergence, RESULTS.successful)
        result = jnp.where(linsolve_fail, state.result, result)
        return terminate, result

    def update_state(self, y, sol, state):
        new_y = (ω(y) + sol.state.descent_dir).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = max_norm((ω(sol.state.descent_dir) / ω(scale)).ω)
        new_state = QNState(
            step=state.step + 1,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=sol.result,
            vector=sentinel,
            operator=sentinel,
        )
        return new_y, new_state
