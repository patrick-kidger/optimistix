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
from typing import Any, cast, Generic

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import AbstractLineSearchState, Aux, Fn, Y
from .._descent import AbstractDescent
from .._minimise import AbstractMinimiser, minimise
from .._misc import max_norm, sum_squares, tree_full_like, tree_inner_prod, tree_where
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .misc import cauchy_termination


def _gradient_step(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    return jnp.array(0.0)


def fletcher_reeves(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    numerator = sum_squares(vector)
    denominator = sum_squares(vector_prev)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def polak_ribiere(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    numerator = tree_inner_prod(vector, (vector**ω - vector_prev**ω).ω)
    denominator = sum_squares(vector_prev)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    beta = cast(
        Scalar, jnp.where(pred, jnp.clip(numerator / safe_denom, a_min=0), jnp.inf)
    )
    return beta


def hestenes_stiefel(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    grad_diff = (vector**ω - vector_prev**ω).ω
    numerator = tree_inner_prod(vector, grad_diff)
    denominator = tree_inner_prod(diff_prev, grad_diff)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


def dai_yuan(vector: Y, vector_prev: Y, diff_prev: Y) -> Scalar:
    numerator = sum_squares(vector)
    denominator = tree_inner_prod(diff_prev, (vector**ω - vector_prev**ω).ω)
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, jnp.inf)


class NonlinearCGDescent(AbstractDescent[Y]):
    """The nonlinear conjugate gradient step.


    This requires the following `options`:
    - `vector`: The gradient of the objective function to minimise.
    - `vector_prev`: The gradient of the objective function to minimise at the
        previous step.
    - `diff_prev`: The `diff` of the previous step.
    """

    method: Callable

    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        vector = options["vector"]
        vector_prev = options["vector_prev"]
        diff_prev = options["diff_prev"]
        beta = lax.cond(
            jnp.isfinite(max_norm(vector_prev)) & jnp.isfinite(max_norm(diff_prev)),
            self.method,
            _gradient_step,
            vector,
            vector_prev,
            diff_prev,
        )
        negative_gradient = (-(vector**ω)).ω
        nonlinear_cg_direction = (negative_gradient**ω + beta * diff_prev**ω).ω
        diff = tree_where(
            tree_inner_prod(vector, nonlinear_cg_direction) < 0,
            nonlinear_cg_direction,
            negative_gradient,
        )
        return (step_size * diff**ω).ω, RESULTS.successful


class _NonlinearCGState(eqx.Module, Generic[Y]):
    step_size: Scalar
    f_val: Scalar
    f_prev: Scalar
    diff: Y
    diff_prev: Y
    vector: Y
    result: RESULTS


class AbstractNonlinearCG(AbstractMinimiser[_NonlinearCGState[Y], Y, Aux]):
    """The nonlinear conjugate gradient method.

    This can be used with all compatible `line_search` and `descent`s.

    ** Attributes:**

    - `rtol`: Relative tolerance for terminating solve.
    - `atol`: Absolute tolerance for terminating solve.
    - `norm`: The norm used to determine the difference between two iterates in the
        convergence criteria. Defaults to `max_norm`.
    - `line_search`: An line-search minimiser which takes a `descent` object. The
        line-search must only require `options` from the list of:
        - "init_step_size"
        - "vector"
        - "vector_prev"
        - "diff"
        - "diff_prev"
        - "f0"
    """

    rtol: float
    atol: float
    norm: Callable
    line_search: AbstractMinimiser[AbstractLineSearchState, Y, Aux]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _NonlinearCGState[Y]:
        del fn, aux_struct
        return _NonlinearCGState(
            step_size=jnp.array(1.0),
            f_val=jnp.array(jnp.inf, f_struct.dtype),
            f_prev=jnp.array(jnp.inf, f_struct.dtype),
            diff=tree_full_like(y, jnp.inf),
            diff_prev=tree_full_like(y, jnp.inf),
            vector=tree_full_like(y, 1.0),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NonlinearCGState[Y],
        tags: frozenset[object],
    ) -> tuple[Y, _NonlinearCGState[Y], Aux]:
        (f_val, aux), new_grad = jax.value_and_grad(fn, has_aux=True)(y, args)
        line_search_options = {
            "init_step_size": state.step_size,
            "vector": new_grad,
            "vector_prev": state.vector,
            "diff": state.diff,
            "diff_prev": state.diff_prev,
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
        new_y = line_sol.value
        result = RESULTS.where(
            line_sol.result == RESULTS.max_steps_reached,
            RESULTS.successful,
            line_sol.result,
        )
        new_state = _NonlinearCGState(
            step_size=line_sol.state.next_init,
            f_val=f_val,
            f_prev=state.f_val,
            diff=(new_y**ω - y**ω).ω,
            diff_prev=state.diff,
            vector=new_grad,
            result=result,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NonlinearCGState[Y],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
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

    def buffers(self, state: _NonlinearCGState[Y]) -> tuple[()]:
        return ()


#
# TODO(raderj): replace the default line search in all of these with a better
# line search algorithm.
#
# NOTE: we choose not to make `method` a private method for each of these
# classes because it simplifies the API that an end-user may want to write
# to create a new NonlinearCG method.
#


class NonlinearCG(AbstractNonlinearCG):
    def __init__(self, rtol, atol, norm=max_norm, method=polak_ribiere):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.line_search = BacktrackingArmijo(
            NonlinearCGDescent(method=method),
            gauss_newton=False,
            decrease_factor=0.5,
            backtrack_slope=0.1,
        )


NonlinearCG.__init__.__doc__ = """**Arguments:**
- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Defaults to `max_norm`.
- `method`: The function which computes `beta` in `NonlinearCG`. Defaults to
    `polak_ribiere`
"""
