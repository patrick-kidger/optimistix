# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (
    Any,
    Callable,
    cast,
    Optional,
    TYPE_CHECKING,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import ArrayLike, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._misc import max_norm
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox.internal import AbstractClassVar


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
    linear_state: Optional[tuple[lx.AuxLinearOperator, PyTree]]
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    step: Scalar


class _NewtonChord(AbstractRootFinder[_NewtonChordState, Y, Out, Aux]):
    rtol: float
    atol: float
    kappa: float = 1e-2
    norm: Callable = max_norm
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    lower: Optional[float] = None
    upper: Optional[float] = None

    _is_newton: AbstractClassVar[bool]

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _NewtonChordState:
        del options, f_struct, aux_struct
        if self._is_newton:
            linear_state = None
        else:
            jac = lx.JacobianLinearOperator(fn, y, args, tags=tags, _has_aux=True)
            jac = lx.linearise(jac)
            linear_state = (jac, self.linear_solver.init(jac, options={}))
        return _NewtonChordState(
            linear_state=linear_state,
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(1.0),
            result=RESULTS.successful,
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonChordState,
        tags: frozenset[object],
    ) -> tuple[Y, _NewtonChordState, Aux]:
        del options
        fx, _ = fn(y, args)
        if self._is_newton:
            jac = lx.JacobianLinearOperator(fn, y, args, tags=tags, _has_aux=True)
            jac = cast(lx.AuxLinearOperator, lx.linearise(jac))
            sol = lx.linear_solve(jac, fx, self.linear_solver, throw=False)
        else:
            jac, linear_state = state.linear_state  # pyright: ignore
            sol = lx.linear_solve(
                jac, fx, self.linear_solver, state=linear_state, throw=False
            )
        diff = sol.value
        new_y = (y**ω - diff**ω).ω
        if self.lower is not None:
            new_y = jnp.clip(new_y, a_min=self.lower)
        if self.upper is not None:
            new_y = jnp.clip(new_y, a_max=self.upper)
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((diff**ω / scale**ω).ω)
        new_state = _NewtonChordState(
            linear_state=state.linear_state,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=RESULTS.promote(sol.result),
            step=state.step + 1,
        )
        return new_y, new_state, jac.aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonChordState,
        tags: frozenset[object],
    ):
        del fn, y, args, options
        at_least_two = state.step >= 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.kappa)
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (
            at_least_two & (small | diverged | converged)  # pyright: ignore
        )
        result = RESULTS.where(
            diverged, RESULTS.nonlinear_divergence, RESULTS.successful
        )
        result = RESULTS.where(linsolve_fail, state.result, result)
        return terminate, result

    def buffers(self, state: _NewtonChordState) -> tuple[()]:
        return ()


class Newton(_NewtonChord):
    _is_newton = True


class Chord(_NewtonChord):
    _is_newton = False
