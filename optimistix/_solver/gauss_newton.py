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
from typing import Any, Generic, Optional, TYPE_CHECKING
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar

from .._base_solver import AbstractHasTol
from .._custom_types import Aux, Fn, NoAuxFn, Out, SearchState, Y
from .._least_squares import AbstractLeastSquaresSolver
from .._misc import (
    cauchy_termination,
    max_norm,
    NoAux,
    sum_squares,
    tree_full_like,
)
from .._search import AbstractDescent, AbstractSearch, DerivativeInfo, newton_step
from .._solution import RESULTS
from .learning_rate import LearningRate


_NewtonDescentState: TypeAlias = tuple[Y, RESULTS]


class NewtonDescent(AbstractDescent[Y, Out, _NewtonDescentState]):
    """Newton descent direction.

    Given a quadratic bowl `x -> x^T Hess x` -- a local quadratic approximation
    to the target function -- this corresponds to moving in the direction of the bottom
    of the bowl. (Which is *not* the same as steepest descent.)

    This is done by solving a linear system of the form `Hess^{-1} grad`.
    """

    norm: Optional[Callable[[PyTree], Scalar]] = None
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def optim_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[Y, RESULTS]:
        # Dummy values of the right shape; unused.
        return y, RESULTS.successful

    def search_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: _NewtonDescentState,
        deriv_info: DerivativeInfo,
    ) -> _NewtonDescentState:
        del fn, y, args, f, state  # only used within a search, not between searches.
        newton, result = newton_step(deriv_info, self.linear_solver)
        if self.norm is not None:
            newton = (newton**ω / self.norm(newton)).ω
        return newton, result

    def descend(
        self,
        step_size: Scalar,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: _NewtonDescentState,
        deriv_info: DerivativeInfo,
    ) -> tuple[Y, RESULTS, _NewtonDescentState]:
        newton, result = state
        return (-step_size * newton**ω).ω, result, state


NewtonDescent.__init__.__doc__ = """**Arguments:**

- `norm`: If passed, then normalise the gradient using this norm. (The returned step
    will have length `step_size` with respect to this norm.) Optimistix includes three
    built-in norms: [`optimistix.max_norm`][], [`optimistix.rms_norm`][], and
    [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to compute the Newton step.
"""


class _GaussNewtonState(eqx.Module, Generic[Y, SearchState]):
    search_state: SearchState
    y_diff: Y
    f_val: Scalar
    f_prev: Scalar
    accept: Bool[Array, ""]
    result: RESULTS


class AbstractGaussNewton(
    AbstractLeastSquaresSolver[Y, Out, Aux, _GaussNewtonState], AbstractHasTol
):
    """Abstract base class for all Gauss-Newton type methods.

    This includes methods such as [`optimistix.GaussNewton`][],
    [`optimistix.LevenbergMarquardt`][], and [`optimistix.Dogleg`][].
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent]
    search: AbstractVar[AbstractSearch]

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GaussNewtonState[Y, Aux]:
        del aux_struct, options
        sum_squares_struct = jax.eval_shape(sum_squares, f_struct)
        init_search_state = self.search.init(self.descent, NoAux(fn), y, args, f_struct)
        return _GaussNewtonState(
            search_state=init_search_state,
            y_diff=tree_full_like(y, jnp.inf),
            f_val=jnp.array(jnp.inf, sum_squares_struct.dtype),
            f_prev=jnp.array(jnp.inf, sum_squares_struct.dtype),
            accept=jnp.array(True),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GaussNewtonState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Y, _GaussNewtonState[Y, Aux], Aux]:
        residual, lin_fn, aux = jax.linearize(
            lambda _y: fn(_y, args), y, has_aux=True  # pyright: ignore
        )
        jac = lx.FunctionLinearOperator(lin_fn, jax.eval_shape(lambda: y), tags)
        f_val = 0.5 * sum_squares(residual)
        deriv_info = DerivativeInfo.ResidualJac(residual, jac)
        y_diff, accept, result, new_search_state = self.search.search(
            self.descent,
            NoAux(fn),
            y,
            args,
            residual,
            state.search_state,
            deriv_info,
            max_steps=256,  # TODO(kidger): expose as an option somewhere?
        )
        new_y = (y**ω + y_diff**ω).ω
        new_state = _GaussNewtonState(
            search_state=new_search_state,
            y_diff=y_diff,
            f_val=f_val,
            f_prev=state.f_val,
            accept=accept,
            result=result,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GaussNewtonState[Y, Aux],
        tags: frozenset[object],
    ):
        return cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            state.y_diff,
            state.f_val,
            state.f_prev,
            state.accept,
            state.result,
        )

    def buffers(self, state: _GaussNewtonState[Y, Aux]) -> tuple[()]:
        return ()


class GaussNewton(AbstractGaussNewton[Y, Out, Aux]):
    """Gauss-Newton algorithm, for solving nonlinear least-squares problems.

    Note that regularised approaches like [`optimistix.LevenbergMarquardt`][] are
    usually preferred instead.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: AbstractDescent
    search: AbstractSearch

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent(linear_solver=linear_solver)
        self.search = LearningRate(1.0)


GaussNewton.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to compute the Newton step.
"""
