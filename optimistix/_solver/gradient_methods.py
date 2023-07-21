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

from .._custom_types import Aux, Fn, Y
from .._descent import AbstractDescent, AbstractLineSearch
from .._minimise import AbstractMinimiser, minimise
from .._misc import AbstractHasTol, max_norm
from .._solution import RESULTS
from .learning_rate import LearningRate
from .misc import cauchy_termination


class SteepestDescent(AbstractDescent[Y]):
    """The descent direction given by locally following the gradient.

    This requires the following `options`:

    - `vector`: The gradient of the objective function to minimise.
    """

    norm: Optional[Callable[[PyTree], Scalar]] = None

    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        vector = options["vector"]
        if self.norm is None:
            diff = vector
        else:
            diff = (vector**ω / self.norm(vector)).ω
        return (-step_size * diff**ω).ω, RESULTS.successful


SteepestDescent.__init__.__doc__ = """**Arguments:**

- `norm`: If passed, then normalise the gradient using this norm. (The returned step
    will have length `step_size` with respect to this norm.) Optimistix includes three
    built-in norms: [`optimistix.max_norm`][], [`optimistix.rms_norm`][], and
    [`optimistix.two_norm`][].
"""


class _GradientDescentState(eqx.Module, Generic[Y, Aux]):
    step_size: Scalar
    y_prev: Y
    f_val: Scalar
    f_prev: Scalar
    result: RESULTS


class AbstractGradientDescent(
    AbstractMinimiser[_GradientDescentState[Y, Aux], Y, Aux], AbstractHasTol
):
    """The gradient descent method for unconstrained minimisation.

    At every step, this algorithm performs a line search along the steepest descent
    direction. You should subclass this to provide it with a particular choice of line
    search. (E.g. [`optimistix.GradientDescent`][] uses a simple learning rate step.)

    The line search can only require `options` from the list of:

    - "init_step_size"
    - "vector"
    - "f0"
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    line_search: AbstractVar[AbstractLineSearch]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GradientDescentState[Y, Aux]:
        del fn, aux_struct
        maxval = jnp.finfo(f_struct.dtype).max
        return _GradientDescentState(
            step_size=jnp.array(1.0),
            y_prev=y,
            f_val=jnp.array(maxval, dtype=f_struct.dtype),
            f_prev=jnp.array(0.5 * maxval, dtype=f_struct.dtype),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GradientDescentState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Y, _GradientDescentState, Aux]:
        (f_val, aux), new_grad = jax.value_and_grad(fn, has_aux=True)(y, args)
        line_search_options = {
            "init_step_size": state.step_size,
            "vector": new_grad,
            "f0": f_val,
        }
        line_sol = minimise(
            fn,
            self.line_search,
            y,
            args,
            line_search_options,
            has_aux=True,
            throw=False,
        )
        result = RESULTS.where(
            line_sol.result == RESULTS.nonlinear_max_steps_reached,
            RESULTS.successful,
            line_sol.result,
        )
        new_state = _GradientDescentState(
            step_size=line_sol.state.next_init,
            y_prev=y,
            f_val=f_val,
            f_prev=state.f_val,
            result=result,
        )
        return line_sol.value, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GradientDescentState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            (y**ω - state.y_prev**ω).ω,
            state.f_val,
            state.f_prev,
            state.result,
        )

    def buffers(self, state: _GradientDescentState[Y, Aux]) -> tuple[()]:
        return ()


class GradientDescent(AbstractGradientDescent):
    """Classic gradient descent with a learning rate `learning_rate`."""

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    line_search: AbstractLineSearch

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
        self.line_search = LearningRate(SteepestDescent(), learning_rate=learning_rate)


GradientDescent.__init__.__doc__ = """**Arguments:**

- `learning_rate`: Specifies a constant learning rate to use at each step.
- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
"""
