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

import functools as ft
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jax import lax
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._line_search import AbstractDescent, AbstractLineSearch, OneDimensionalFunction
from .._minimise import AbstractMinimiser, minimise
from .._misc import max_norm, tree_full, tree_zeros, tree_zeros_like
from .._solution import RESULTS
from .descent import UnnormalisedGradient
from .learning_rate import LearningRate
from .nonlinear_cg_descent import hestenes_stiefel, NonlinearCGDescent


class _GradOnlyState(eqx.Module):
    descent_state: PyTree
    vector: PyTree[Array]
    operator: lx.AbstractLinearOperator
    diff: PyTree[Array]
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    f_val: PyTree[Array]
    f_prev: PyTree[Array]
    next_init: Array
    aux: Any
    step: Scalar


# note that this is called "GradOnly" and not "VecOnly," despite us often referring
# to the gradient and residual vectors by the name `vector`.
# This is because it doesn't make sense to use this in the least squares setting
# where `vector` is the residual vector, so we know we are always dealing with
# gradients.
class AbstractGradOnly(AbstractMinimiser[_GradOnlyState, Y, Aux]):
    rtol: float
    atol: float
    line_search: AbstractLineSearch
    descent: AbstractDescent
    norm: Callable = max_norm

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: Any,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GradOnlyState:
        f0 = tree_full(f_struct, jnp.inf)
        aux = tree_zeros(aux_struct)
        y_zeros = tree_zeros_like(y)
        operator = lx.IdentityLinearOperator(jax.eval_shape(lambda: y))
        descent_state = self.descent.init_state(
            fn, y, y_zeros, operator, None, args, options
        )
        return _GradOnlyState(
            descent_state=descent_state,
            vector=y_zeros,
            operator=operator,
            diff=jtu.tree_map(lambda x: jnp.full(x.shape, jnp.inf), y),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=RESULTS.successful,
            f_val=f0,
            f_prev=f0,
            next_init=jnp.array(1.0),
            aux=aux,
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: Any,
        options: dict[str, Any],
        state: _GradOnlyState,
        tags: frozenset[object],
    ) -> tuple[Y, _GradOnlyState, Aux]:
        def main_pass(y, state):
            descent = eqx.Partial(
                self.descent,
                descent_state=state.descent_state,
                args=args,
                options=options,
            )
            problem_1d = OneDimensionalFunction(fn, descent, y)
            # Line search should compute`f0` the first time it is called.
            # Since `first_pass` will be computed at step 0 of this solver, the
            # first time the line search is called is step 1. Each time after this,
            # we pass `f0` via these `line_search_options`.
            line_search_options = {
                "f0": state.f_val,
                "compute_f0": (state.step == 1),
                "vector": state.vector,
                "operator": state.operator,
                "diff": y,
            }

            line_search_options["predicted_reduction"] = ft.partial(
                self.descent.predicted_reduction,
                descent_state=state.descent_state,
                args=args,
                options={},
            )
            init = jnp.where(
                state.step == 0,
                self.line_search.first_init(state.vector, state.operator, options),
                state.next_init,
            )
            line_sol = minimise(
                fn=problem_1d,
                has_aux=True,
                solver=self.line_search,
                y0=init,
                args=args,
                options=line_search_options,
                max_steps=100,
                throw=False,
            )
            (f_val, diff, new_aux, _, next_init) = line_sol.aux
            return f_val, diff, new_aux, line_sol.result, next_init

        def first_pass(y, state):
            return (
                jnp.inf,
                ω(y).call(jnp.zeros_like).ω,
                state.aux,
                RESULTS.successful,
                state.next_init,
            )

        # this lax.cond allows us to avoid an extra compilation of f(y) in the init.
        f_val, diff, new_aux, result, next_init = lax.cond(
            state.step == 0, first_pass, main_pass, y, state
        )
        new_y = (y**ω + diff**ω).ω
        new_grad, _ = jax.jacrev(fn, has_aux=True)(new_y, args)
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(diff) / ω(scale)).ω)
        descent_state = self.descent.update_state(
            state.descent_state, diff, new_grad, state.operator, None, options
        )
        result = RESULTS.where(
            result == RESULTS.max_steps_reached, RESULTS.successful, result
        )
        new_state = _GradOnlyState(
            descent_state=descent_state,
            vector=new_grad,
            operator=state.operator,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=result,
            f_val=f_val,
            f_prev=state.f_val,
            next_init=next_init,
            aux=new_aux,
            step=state.step + 1,
        )
        return new_y, new_state, new_aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: Any,
        options: dict[str, Any],
        state: _GradOnlyState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        at_least_two = state.step >= 2
        f_diff = jnp.abs(state.f_val - state.f_prev)
        converged = f_diff < self.rtol * jnp.abs(state.f_prev) + self.atol
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (at_least_two & converged)
        result = RESULTS.where(linsolve_fail, state.result, RESULTS.successful)
        return terminate, result

    def buffers(self, state: _GradOnlyState) -> tuple[()]:
        return ()


class GradOnly(AbstractGradOnly):
    pass


class GradientDescent(AbstractGradOnly):
    def __init__(
        self,
        rtol: float,
        atol: float,
        learning_rate: float,
        norm: Callable = max_norm,
    ):
        self.rtol = rtol
        self.atol = atol
        self.line_search = LearningRate(learning_rate)
        self.descent = UnnormalisedGradient()
        self.norm = norm


class NonlinearCG(AbstractGradOnly):
    def __init__(
        self,
        rtol: float,
        atol: float,
        line_search: AbstractLineSearch,
        norm: Callable = max_norm,
        method: Callable = hestenes_stiefel,
    ):
        self.rtol = rtol
        self.atol = atol
        self.line_search = line_search
        self.descent = NonlinearCGDescent(method)
        self.norm = norm