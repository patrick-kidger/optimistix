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

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar

from .._base_solver import AbstractHasTol
from .._custom_types import Aux, Fn, NoAuxFn, Out, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import cauchy_termination, max_norm, NoAux, tree_full_like
from .._search import AbstractDescent, AbstractSearch, DerivativeInfo
from .._solution import RESULTS
from .learning_rate import LearningRate


class SteepestDescent(AbstractDescent[Y, Out, None]):
    """The descent direction given by locally following the gradient."""

    norm: Optional[Callable[[PyTree], Scalar]] = None

    def optim_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        return None

    def search_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: None,
        deriv_info: DerivativeInfo,
    ) -> None:
        return None

    def descend(
        self,
        step_size: Scalar,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: None,
        deriv_info: DerivativeInfo,
    ) -> tuple[Y, RESULTS, None]:
        assert isinstance(
            deriv_info,
            (
                DerivativeInfo.Grad,
                DerivativeInfo.GradHessian,
                DerivativeInfo.GradHessianInv,
                DerivativeInfo.ResidualJac,
            ),
        )
        grad = deriv_info.grad
        if self.norm is not None:
            grad = (grad**ω / self.norm(grad)).ω
        return (-step_size * grad**ω).ω, RESULTS.successful, None


SteepestDescent.__init__.__doc__ = """**Arguments:**

- `norm`: If passed, then normalise the gradient using this norm. (The returned step
    will have length `step_size` with respect to this norm.) Optimistix includes three
    built-in norms: [`optimistix.max_norm`][], [`optimistix.rms_norm`][], and
    [`optimistix.two_norm`][].
"""


class _GradientDescentState(eqx.Module, Generic[Y, SearchState]):
    search_state: SearchState
    y_diff: Y
    f_val: Scalar
    f_prev: Scalar
    accept: Bool[Array, ""]
    result: RESULTS


class AbstractGradientDescent(
    AbstractMinimiser[Y, Aux, _GradientDescentState], AbstractHasTol
):
    """The gradient descent method for unconstrained minimisation.

    At every step, this algorithm performs a line search along the steepest descent
    direction. You should subclass this to provide it with a particular choice of line
    search. (E.g. [`optimistix.GradientDescent`][] uses a simple learning rate step.)
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent]
    search: AbstractVar[AbstractSearch]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GradientDescentState:
        del aux_struct
        maxval = jnp.finfo(f_struct.dtype).max
        init_search_state = self.search.init(self.descent, NoAux(fn), y, args, f_struct)
        return _GradientDescentState(
            search_state=init_search_state,
            y_diff=tree_full_like(y, jnp.inf),
            f_val=jnp.array(maxval, dtype=f_struct.dtype),
            f_prev=jnp.array(0.5 * maxval, dtype=f_struct.dtype),
            accept=jnp.array(True),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GradientDescentState,
        tags: frozenset[object],
    ) -> tuple[Y, _GradientDescentState, Aux]:
        f_val: Scalar
        aux: Aux
        grad: Y
        (f_val, aux), grad = jax.value_and_grad(fn, has_aux=True)(y, args)
        deriv_info = DerivativeInfo.Grad(grad)
        y_diff, accept, result, new_search_state = self.search.search(
            self.descent,
            NoAux(fn),
            y,
            args,
            f_val,
            state.search_state,
            deriv_info,
            max_steps=256,  # TODO(kidger): offer an API for this?
        )
        new_y = (y**ω + y_diff**ω).ω
        new_state = _GradientDescentState(
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
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GradientDescentState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
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

    def buffers(self, state: _GradientDescentState[Y, Aux]) -> tuple[()]:
        return ()


class GradientDescent(AbstractGradientDescent[Y, Aux]):
    """Classic gradient descent with a learning rate `learning_rate`."""

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: AbstractDescent
    search: AbstractSearch

    def __init__(
        self,
        learning_rate: float,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = SteepestDescent()
        self.search = LearningRate(learning_rate)


GradientDescent.__init__.__doc__ = """**Arguments:**

- `learning_rate`: Specifies a constant learning rate to use at each step.
- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
"""
