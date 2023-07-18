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
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from .._custom_types import AbstractLineSearchState, Args, Aux, Fn, Out, Y
from .._descent import AbstractDescent
from .._least_squares import AbstractLeastSquaresSolver
from .._minimise import AbstractMinimiser, minimise
from .._misc import (
    max_norm,
    sum_squares,
    tree_full_like,
    two_norm,
)
from .._solution import RESULTS
from .learning_rate import LearningRate
from .misc import cauchy_termination


def _is_array_or_struct(x):
    return eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)


class NewtonDescent(AbstractDescent[Y]):
    normalise: bool = False
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        vector = options["vector"]
        try:
            operator_inv = options["operator_inv"]
        except KeyError:
            operator_inv = None
        try:
            operator = options["operator"]
        except KeyError:
            operator = None

        if operator_inv is not None:
            newton = operator_inv.mv(vector)
            result = RESULTS.successful
        elif operator is not None:
            out = lx.linear_solve(
                operator,
                vector,
                self.linear_solver,
            )
            newton = out.value
            result = RESULTS.promote(out.result)
        else:
            raise ValueError(
                "At least one of `operator` or `operator_inv` must be "
                "passed to `NewtonDescent` via `options`."
            )
        if self.normalise:
            diff = (newton**ω / two_norm(newton)).ω
        else:
            diff = newton
        return (-step_size * diff**ω).ω, result


class _GaussNewtonState(eqx.Module, Generic[Y, Aux]):
    step_size: Scalar
    diff: Y
    f_val: Scalar
    f_prev: Scalar
    result: RESULTS


def _minimise_fn(fn: Fn[Y, Out, Aux], y: Y, args: Args) -> tuple[Scalar, Aux]:
    residual, aux = fn(y, args)
    return sum_squares(residual), aux


class AbstractGaussNewton(
    AbstractLeastSquaresSolver[_GaussNewtonState[Y, Aux], Y, Out, Aux]
):
    rtol: float
    atol: float
    norm: Callable
    line_search: AbstractMinimiser[AbstractLineSearchState, Y, Aux]

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
        return _GaussNewtonState(
            step_size=jnp.array(1.0),
            diff=tree_full_like(y, jnp.inf),
            f_val=jnp.array(jnp.inf, sum_squares_struct.dtype),
            f_prev=jnp.array(jnp.inf, sum_squares_struct.dtype),
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
        in_structure = jax.eval_shape(lambda: y)
        new_operator = lx.FunctionLinearOperator(lin_fn, in_structure)
        f_val = sum_squares(residual)

        # TODO(raderj): remove the `fn` and `y` options.
        line_search_options = {
            "init_step_size": state.step_size,
            "vector": residual,
            "operator": new_operator,
            "operator_inv": None,
            "f0": f_val,
            "fn": fn,
            "y": y,
        }
        line_sol = minimise(
            eqx.Partial(_minimise_fn, fn),
            self.line_search,
            y,
            args,
            line_search_options,
            has_aux=True,
            throw=False,
        )
        new_y = line_sol.value
        result = RESULTS.where(
            line_sol.result == RESULTS.max_steps_reached,
            RESULTS.successful,
            line_sol.result,
        )
        new_state = _GaussNewtonState(
            step_size=line_sol.state.next_init,
            diff=(new_y**ω - y**ω).ω,
            f_val=f_val,
            f_prev=state.f_val,
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
            state.diff,
            state.f_val,
            state.f_prev,
            state.result,
        )

    def buffers(self, state: _GaussNewtonState[Y, Aux]) -> tuple[()]:
        return ()


class GaussNewton(AbstractGaussNewton):
    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.line_search = LearningRate(
            NewtonDescent(linear_solver=linear_solver), jnp.array(1.0)
        )
