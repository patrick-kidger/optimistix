import abc
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar

from ..misc import max_norm
from ..root_find import AbstractRootFinder, RootFindProblem
from ..solution import RESULTS


def _small(diffsize: Scalar) -> Bool[ArrayLike, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[ArrayLike, " "]:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: float) -> Bool[ArrayLike, " "]:
    return (factor > 0) & (factor < tol)


class _NewtonChordState(eqx.Module):
    linear_state: Optional[PyTree]
    step: Scalar
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS


class _NewtonChord(AbstractRootFinder):
    rtol: float
    atol: float
    kappa: float = 1e-2
    norm: Callable = max_norm
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    modify_jac: Callable[
        [lx.JacobianLinearOperator], lx.AbstractLinearOperator
    ] = lx.linearise
    lower: float = None
    upper: float = None

    @property
    @abc.abstractmethod
    def _is_newton(self) -> bool:
        ...

    def init(
        self,
        problem: RootFindProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
    ):
        del options, f_struct, aux_struct
        if self._is_newton:
            linear_state = None
        else:
            jac = lx.JacobianLinearOperator(
                problem.fn, y, args, tags=problem.tags, _has_aux=True
            )
            jac = self.modify_jac(jac)
            linear_state = (jac, self.linear_solver.init(jac, options={}))
        return _NewtonChordState(
            linear_state=linear_state,
            step=jnp.array(0),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=jnp.array(RESULTS.successful),
        )

    def step(
        self,
        problem: RootFindProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: _NewtonChordState,
    ):
        del options
        fx, _ = problem.fn(y, args)
        if self._is_newton:
            jac = lx.JacobianLinearOperator(
                problem.fn, y, args, tags=problem.tags, _has_aux=True
            )
            jac = self.modify_jac(jac)
            sol = lx.linear_solve(jac, fx, self.linear_solver, throw=False)
        else:
            jac, linear_state = state.linear_state
            sol = lx.linear_solve(
                jac, fx, self.linear_solver, state=linear_state, throw=False
            )
        diff = sol.value
        new_y = (y**ω - diff**ω).ω
        # this clip is very important for LM, as it keeps lambda > 0.
        if self.lower is not None:
            new_y = jnp.clip(new_y, a_min=self.lower)
        if self.upper is not None:
            new_y = jnp.clip(new_y, a_max=self.upper)
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

    def terminate(
        self,
        problem: RootFindProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: _NewtonChordState,
    ):
        del problem, y, args, options
        at_least_two = state.step >= 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.kappa)
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (at_least_two & (small | diverged | converged))
        result = jnp.where(diverged, RESULTS.nonlinear_divergence, RESULTS.successful)
        result = jnp.where(linsolve_fail, state.result, result)
        return terminate, result

    def buffers(self, state):
        return ()


class Newton(_NewtonChord):
    _is_newton = True


class Chord(_NewtonChord):
    _is_newton = False
