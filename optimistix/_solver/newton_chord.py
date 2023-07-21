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
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._misc import AbstractHasTol, max_norm, tree_full_like
from .._root_find import AbstractRootFinder
from .._solution import RESULTS
from .misc import cauchy_termination


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


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
    linear_state: Optional[tuple[lx.AbstractLinearOperator, PyTree]]
    diff: Y
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    step: Scalar


class _NewtonChord(
    AbstractRootFinder[Y, Out, Aux, _NewtonChordState[Y]], AbstractHasTol
):
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    kappa: float = 1e-2
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
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
        else:
            f_val = None
        return _NewtonChordState(
            f_val=f_val,
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
        lower = options.get("lower")
        upper = options.get("upper")
        del options
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
        if lower is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, a_min=b), new_y, lower)
        if upper is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, a_max=b), new_y, upper)
        if lower is not None or upper is not None:
            diff = (y**ω - new_y**ω).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((diff**ω / scale**ω).ω)
        if self.cauchy_termination:
            f_val = fx
        else:
            f_val = None
        new_state = _NewtonChordState(
            f_val=f_val,
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
            # Compare `f_val` against 0, not against some `f_prev`. This is because
            # we're doing a root-find and know that we're aiming to get close to zero.
            # Note that this does mean that the `rtol` is ignored in f-space, and only
            # `atol` matters.
            terminate, result = cauchy_termination(
                self.rtol,
                self.atol,
                self.norm,
                y,
                state.diff,
                state.f_val,
                jtu.tree_map(jnp.zeros_like, state.f_val),
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
    """Newton's method for root finding. Also sometimes known as Newton--Raphson.

    Unlike the SciPy implementation of Newton's method, the Optimistix version also
    works for vector-valued (or PyTree-valued) `y`.

    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    """

    _is_newton = True


class Chord(_NewtonChord):
    """The Chord method of root finding.

    This is equivalent to the Newton method, except that the Jacobian is computed only
    once at the initial point `y0`, and then reused throughout the computation. This is
    a useful way to cheapen the solve, if `y0` is expected to be a good initial guess
    and the target function does not change too rapidly. (For example this is the
    standard technique used in implicit Runge--Kutta methods, when solving differential
    equations.)

    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    """

    _is_newton = False


_init_doc = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][]. Only used when
    `cauchy_termination=True`.
- `kappa`: A tolerance for early convergence check when `cauchy_termination=False`.
- `linear_solver`: The linear solver used to compute the Newton step.
- `cauchy_termination`: When `True`, use the Cauchy termination condition, that
    two adjacent iterates should have a small difference between them. This is usually
    the standard choice when solving general root finding problems. When `False`, use
    a procedure which attempts to detect slow convergence, and quickly fail the solve
    if so. This is useful when iteratively performing the root-find, refining the
    target problem for those which fail. This comes up when solving differential
    equations with adaptive step sizing and implicit solvers. The exact procedure is as
    described in Section IV.8 of Hairer & Wanner, "Solving Ordinary Differential
    Equations II".
"""

Newton.__init__.__doc__ = _init_doc
Chord.__init__.__doc__ = _init_doc
