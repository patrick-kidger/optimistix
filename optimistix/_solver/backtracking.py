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

from typing import Any, Generic, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar, ScalarLike

from .._custom_types import DescentState, NoAuxFn, Out, Y
from .._misc import (
    sum_squares,
    tree_dot,
    tree_full_like,
    tree_where,
)
from .._search import AbstractDescent, AbstractSearch, DerivativeInfo
from .._solution import RESULTS


class _BacktrackingCarry(eqx.Module, Generic[Y, DescentState]):
    step_size: Scalar
    f_best: Scalar
    y_diff: Y
    result: RESULTS
    state: DescentState


class BacktrackingArmijo(AbstractSearch[Y, Out, DescentState]):
    """Perform a backtracking Armijo line search."""

    decrease_factor: ScalarLike = 0.5
    slope: ScalarLike = 0.1
    step_init: ScalarLike = 1.0

    def __post_init__(self):
        self.decrease_factor = eqx.error_if(
            self.decrease_factor,
            (self.decrease_factor <= 0)  # pyright: ignore
            | (self.decrease_factor >= 1),  # pyright: ignore
            "`BacktrackingArmoji(decrease_factor=...)` must be between 0 and 1.",
        )
        self.slope = eqx.error_if(
            self.slope,
            (self.slope <= 0) | (self.slope >= 1),  # pyright: ignore
            "`BacktrackingArmoji(slope=...)` must be between 0 and 1.",
        )
        self.step_init = eqx.error_if(
            self.step_init,
            self.step_init <= 0,  # pyright: ignore
            "`BacktrackingArmoji(step_init=...)` must be strictly greater than 0.",
        )

    def init(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Scalar],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> DescentState:
        return descent.optim_init(fn, y, args, f_struct)

    def search(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: DescentState,
        deriv_info: DerivativeInfo,
        max_steps: Optional[int],
    ) -> tuple[Y, Bool[Array, ""], RESULTS, DescentState]:
        state = descent.search_init(fn, y, args, f, state, deriv_info)
        if isinstance(
            deriv_info,
            (
                DerivativeInfo.Grad,
                DerivativeInfo.GradHessian,
                DerivativeInfo.GradHessianInv,
            ),
        ):
            min_fn = fn
            min_f0 = f
        elif isinstance(deriv_info, DerivativeInfo.ResidualJac):
            min_fn = lambda y, args: 0.5 * sum_squares(fn(y, args))
            min_f0 = 0.5 * sum_squares(f)
        else:
            assert False
        grad = deriv_info.grad
        assert isinstance(min_f0, Scalar)

        def cond_fun(carry: _BacktrackingCarry):
            """Terminate when the Armijo condition is satisfied. That is, `fn(new_y)`
            must do better than its linear approximation:
            `fn(y0 + y_diff) < fn(y0) + grad•y_diff`
            """
            # This should maybe include a Cauchy condition as well, especially if we
            # pass `step_size=0`
            failure = carry.result != RESULTS.successful
            predicted_reduction = tree_dot(grad, carry.y_diff)
            satisfies_armijo = carry.f_best <= min_f0 + self.slope * predicted_reduction
            has_reduction = predicted_reduction <= 0
            return failure | (satisfies_armijo & has_reduction)

        def body_fun(carry: _BacktrackingCarry) -> _BacktrackingCarry:
            y_diff, result, new_state = descent.descend(
                carry.step_size, fn, y, args, f, carry.state, deriv_info
            )
            y_candidate = (y**ω + y_diff**ω).ω
            min_f = min_fn(y_candidate, args)
            assert isinstance(min_f, Scalar)
            new_best = min_f < carry.f_best
            new_step_size = self.decrease_factor * carry.step_size
            new_f = jnp.where(new_best, f, carry.f_best)
            new_y_diff = tree_where(new_best, y_diff, carry.y_diff)
            new_carry = _BacktrackingCarry(
                step_size=new_step_size,
                f_best=new_f,
                y_diff=new_y_diff,
                result=result,
                state=new_state,
            )
            return new_carry

        init_carry = _BacktrackingCarry(
            step_size=jnp.asarray(self.step_init),
            f_best=min_f0,
            y_diff=tree_full_like(y, 0),
            result=RESULTS.successful,
            state=state,
        )
        final_carry = eqxi.while_loop(
            cond_fun, body_fun, init_carry, kind="checkpointed", max_steps=max_steps
        )

        return (
            final_carry.y_diff,
            jnp.array(True),
            final_carry.result,
            final_carry.state,
        )


BacktrackingArmijo.__init__.__doc__ = """**Arguments:**

- `decrease_factor`: The rate at which to backtrack, i.e.
    `next_stepsize = decrease_factor * current_stepsize`. Must be between 0 and 1.
- `slope`: The slope of of the linear approximation to
    `f` that the backtracking algorithm must exceed to terminate. Larger
    means stricter termination criteria. Must be between 0 and 1.
- `step_init`: The first `step_size` the backtracking algorithm will
    try. Must be greater than 0.
"""
