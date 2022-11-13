import abc
from typing import Callable, Optional

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import PyTree

from ..custom_types import Scalar
from ..linear_operator import JacobianLinearOperator
from ..linear_solve import AbstractLinearSolver, AutoLinearSolver, linear_solve
from ..misc import rms_norm
from ..results import RESULTS
from ..root_finding import AbstractRootFindSolver


def _small(diffsize: Scalar) -> bool:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> bool:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: Scalar) -> bool:
    return (factor > 0) & (factor < tol)


class _NewtonChordState(eqx.Module):
    linear_state: Optional[PyTree]
    step: Scalar
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS


class _NewtonChord(AbstractRootFindSolver):
    rtol: float
    atol: float
    kappa: float = 1e-2
    norm: Callable = rms_norm
    linear_solver: AbstractLinearSolver = AutoLinearSolver()

    @property
    @abc.abstractmethod
    def _is_newton(self) -> bool:
        ...

    def init(self, root_prob, y, args, options):
        del options
        if self._is_newton:
            linear_state = None
        else:
            jac = JacobianLinearOperator(
                root_prob.fn, y, args, pattern=root_prob.pattern, has_aux=True
            )
            if self.linear_solver.will_materialise():
                jac = jac.materialise()
            else:
                jac = jac.linearise()
            linear_state = (jac, self.linear_solver.init(jac))
        return _NewtonChordState(
            linear_state=linear_state,
            step=jnp.array(0),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=jnp.array(RESULTS.successful),
        )

    def step(self, root_prob, y, args, options, state):
        del options
        fx = root_prob.fn(y, args)
        if self._is_newton:
            jac = JacobianLinearOperator(
                root_prob.fn, y, args, pattern=root_prob.pattern
            )
            if self.linear_solver.will_materialise():
                jac = jac.materialise()
            else:
                jac = jac.linearise()
            sol = linear_solve(jac, fx, self.linear_solver, throw=False)
        else:
            jac, state = state.linear_state
            sol = linear_solve(jac, fx, self.linear_solver, state=state, throw=False)
        diff = sol.value
        new_y = (y**ω - diff**ω).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((diff**ω / scale**ω).ω)
        new_state = _NewtonChordState(
            linear_state=state.linear_state,
            step=state.step + 1,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=sol.result,
        )
        return new_y, new_state, jac.aux

    def terminate(self, root_prob, y, args, options, state):
        del root_prob, y, args, options
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


class Newton(_NewtonChord):
    _is_newton = True


class Chord(_NewtonChord):
    _is_newton = False
