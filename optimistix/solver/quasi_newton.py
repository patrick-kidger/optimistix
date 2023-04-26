import abc
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike, Bool, PyTree

from ..custom_types import Scalar
from ..line_search import AbstractDescent, AbstractLineSearch
from ..linear_operator import AbstractLinearOperator
from ..minimise import AbstractMinimiser
from ..solution import RESULTS


def _small(diffsize: Scalar) -> Bool[ArrayLike, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[ArrayLike, " "]:
    return jnp.invert(jnp.isfinite(rate))


def _converged(factor: Scalar, tol: Scalar) -> Bool[ArrayLike, " "]:
    return (factor > 0) & (factor < tol)


class QNState(eqx.Module):
    step: Scalar
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    vector: Optional[PyTree[ArrayLike]]
    operator: Optional[AbstractLinearOperator]
    aux: Any


class AbstractQuasiNewton(AbstractMinimiser):
    line_search: AbstractLineSearch
    descent: AbstractDescent
    converged_tol: float

    @abc.abstractmethod
    def init(self, problem, y, args, options):
        ...

    @abc.abstractmethod
    def step(self, problem, y, args, options, state):
        ...

    def terminate(self, problem, y, args, options, state):
        at_least_two = state.step >= 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.converged_tol)
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (at_least_two & (small | diverged | converged))
        result = jnp.where(diverged, RESULTS.nonlinear_divergence, RESULTS.successful)
        result = jnp.where(linsolve_fail, state.result, result)
        return terminate, result

    @abc.abstractmethod
    def buffer(self, state):
        ...
