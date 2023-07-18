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

from collections.abc import Callable
from typing import Any, cast, Generic, Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._misc import max_norm, tree_full_like
from .._root_find import AbstractRootFinder
from .._solution import RESULTS
from .misc import cauchy_termination


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox.internal import AbstractClassVar


def _small(diffsize: Scalar) -> Bool[Array, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[Array, " "]:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: float) -> Bool[Array, " "]:
    return (factor > 0) & (factor < tol)


class _NewtonChordState(eqx.Module, Generic[Y]):
    f_val: PyTree[Array]
    f_prev: PyTree[Array]
    linear_state: Optional[tuple[lx.AuxLinearOperator, PyTree]]
    diff: Y
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    step: Scalar


class _NewtonChord(AbstractRootFinder[_NewtonChordState[Y], Y, Out, Aux]):
    rtol: float
    atol: float
    kappa: float = 1e-2
    norm: Callable = max_norm
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    lower: Optional[float] = None
    upper: Optional[float] = None
    cauchy_termination: bool = True

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
    ) -> _NewtonChordState[Y]:
        del options, aux_struct
        if self._is_newton:
            linear_state = None
        else:
            # TODO(kidger): evaluate on just the first step, to reduce compile time.
            jac = lx.JacobianLinearOperator(fn, y, args, tags=tags, _has_aux=True)
            jac = lx.linearise(jac)
            linear_state = (jac, self.linear_solver.init(jac, options={}))
        if self.cauchy_termination:
            f_val = tree_full_like(f_struct, jnp.inf)
            f_prev = tree_full_like(f_struct, 0)
        else:
            f_val = None
            f_prev = None
        return _NewtonChordState(
            f_val=f_val,
            f_prev=f_prev,
            linear_state=linear_state,
            diff=tree_full_like(y, jnp.inf),
            diffsize=jnp.array(jnp.inf),
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
        state: _NewtonChordState[Y],
        tags: frozenset[object],
    ) -> tuple[Y, _NewtonChordState[Y], Aux]:
        del options
        fx, _ = fn(y, args)
        if self._is_newton:
            fx, lin_fn, aux = jax.linearize(lambda _y: fn(_y, args), y, has_aux=True)
            jac = lx.FunctionLinearOperator(
                lin_fn, jax.eval_shape(lambda: y), tags=tags
            )
            sol = lx.linear_solve(jac, fx, self.linear_solver, throw=False)
        else:
            fx, aux = fn(y, args)
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
        if self.cauchy_termination:
            f_val = fx
        else:
            f_val = None
        new_state = _NewtonChordState(
            f_val=f_val,
            f_prev=state.f_val,
            linear_state=state.linear_state,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=RESULTS.promote(sol.result),
            step=state.step + 1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonChordState[Y],
        tags: frozenset[object],
    ):
        del fn, args, options
        linsolve_fail = state.result != RESULTS.successful
        linsolve_fail = cast(Array, linsolve_fail)
        if self.cauchy_termination:
            terminate, result = cauchy_termination(
                self.rtol,
                self.atol,
                self.norm,
                y,
                state.diff,
                state.f_val,
                state.f_prev,
                state.result,
            )
        else:
            # TODO(kidger): perform only one iteration when solving a linear system!
            at_least_two = state.step >= 2
            rate = state.diffsize / state.diffsize_prev
            factor = state.diffsize * rate / (1 - rate)
            small = _small(state.diffsize)
            diverged = _diverged(rate)
            converged = _converged(factor, self.kappa)
            terminate = at_least_two & (small | diverged | converged)
            result = RESULTS.where(
                diverged, RESULTS.nonlinear_divergence, RESULTS.successful
            )
        result = RESULTS.where(linsolve_fail, state.result, result)
        terminate = linsolve_fail | terminate
        return terminate, result

    def buffers(self, state: _NewtonChordState) -> tuple[()]:
        return ()


class Newton(_NewtonChord):
    """Newton's method of root finding."""

    _is_newton = True


class Chord(_NewtonChord):
    """The chord method of root finding."""

    _is_newton = False

    rtol: float
    atol: float
    kappa: float = 1e-2
    norm: Callable = max_norm
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    lower: Optional[float] = None
    upper: Optional[float] = None
    cauchy_termination: bool = True


_init_doc = """**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `kappa`: A tolerance for early convergence check when `cauchy_termination=False`.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Defaults to `max_norm`.
- `linear_solver`: The linear solver used to compute the Newton step. Defaults to
    `lx.AutoLinearSolver(well_posed=None)`.
- `lower`: The lowe bound on the interval which contains the root.
- `upper`: The upper bound on the interval which contains the root.
- `cauchy_termination`: When `True`, use a cauchy-like termination condition. When
    `False`, use a condition more likely to terminate early.
"""

Newton.__init__.__doc__ = _init_doc
Chord.__init__.__doc__ = _init_doc
