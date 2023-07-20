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

from typing import Any, cast, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, Int, PyTree, Scalar

from .._custom_types import (
    AbstractLineSearchState,
    Aux,
    Fn,
    sentinel,
    Y,
)
from .._descent import AbstractDescent, AbstractLineSearch
from .._misc import (
    is_linear,
    tree_full_like,
    tree_inner_prod,
    tree_where,
)
from .._solution import RESULTS


def _descent_no_results(descent, args, options, step_size):
    diff, results = descent(step_size, args, options)
    del results
    return diff


class _BacktrackingState(AbstractLineSearchState, Generic[Y]):
    next_init: Scalar
    step_size: Scalar
    current_y: Y
    current_f: Scalar
    best_f_val: Scalar
    diff: Y
    cached_diff: Y
    grad: Y
    result: RESULTS
    step: Int[Array, ""]


class BacktrackingArmijo(AbstractLineSearch[_BacktrackingState[Y], Y, Aux]):
    """Compute `y_new` from `y` using backtracking Armijo line search.

    This requires the following to be passed via `options`:

    - `f0`: The value of the function at the initial point `y`. (This could be computed
        by making another function evaluation inside the line search -- but this would
        increase runtime and compilation time, and it's usually already available from
        the caller.)
    - `init_step_size`: The initial `step_size` that the line search will try.
    - `vector`: The residual vector if `gauss_newton=True`, the gradient vector
        otherwise.
    - `operator`: Only necessary when `gauss_newton=True`. The Jacobian operator of the
        function in a least-squares problem.
    """

    descent: AbstractDescent[Y]
    gauss_newton: bool
    decrease_factor: float = 0.5
    backtrack_slope: float = 0.1
    backtracking_init: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=1.0)

    def __post_init__(self):
        eps = jnp.finfo(jnp.float64).eps
        if self.decrease_factor < eps:
            raise ValueError(
                "`decrease_factor` of backtracking line search must be greater than 0."
            )
        if self.backtracking_init < eps:
            raise ValueError(
                "`backtracking_init` of backtracking line search must be greater than "
                "0."
            )
        if self.backtrack_slope < 0:
            raise ValueError(
                "`backtrack_slope` of backtracking line search must be greater than 0."
            )

        if self.backtrack_slope > 1:
            raise ValueError(
                "`backtrack_slope` of backtracking line search must be less than 1."
            )

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BacktrackingState[Y]:
        del aux_struct, tags
        f0 = options["f0"]
        vector = options["vector"]
        diff = tree_full_like(y, jnp.inf)
        step_size = options["init_step_size"]
        if self.gauss_newton:
            # Backtracking line search uses the gradient of `f` in the termination
            # condition. When we are solving a Gauss-Newton problem, we expect
            # `vector` to be a vector of residuals `r`, and operator to be the
            # Jacobian `J`. In this case, we can compute the gradient as
            # `J^T r`.
            operator = options["operator"]
            grad = operator.transpose().mv(vector)
        else:
            # When we are not in the Gauss-Newton case, we expect `vector` to be
            # the gradient directly, so we don't need to do anything :).
            grad = vector
        # If the descent computes `step_size * diff` for a fixed vector `diff`
        # that isn't dependent upon `step_size`, then we can cache `diff`
        # and avoid recomputing it. In other words, we do a classical line search.
        if is_linear(
            eqx.Partial(_descent_no_results, self.descent, args, options),
            jnp.array(1.0),
            output=y,
        ):
            cached_diff, result = self.descent(jnp.array(1.0), args, options)
        else:
            cached_diff = sentinel
            result = RESULTS.successful
        return _BacktrackingState(
            next_init=step_size,
            step_size=step_size,
            current_y=y,
            current_f=f0,
            best_f_val=f0,
            diff=diff,
            cached_diff=cached_diff,
            grad=grad,
            result=result,
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BacktrackingState[Y],
        tags: frozenset[object],
    ) -> tuple[Y, _BacktrackingState[Y], Aux]:
        del tags
        if state.cached_diff is sentinel:
            diff, result = self.descent(state.step_size, args, options)
        else:
            diff = (state.step_size * state.cached_diff**ω).ω
            result = state.result
        proposed_y = (state.current_y**ω + diff**ω).ω
        f_val, aux = fn(proposed_y, args)
        new_min = f_val < state.best_f_val
        new_y = tree_where(new_min, proposed_y, y)
        best_f_val = jnp.where(new_min, f_val, state.best_f_val)
        new_step_size = self.decrease_factor * state.step_size
        new_state = _BacktrackingState(
            next_init=self.backtracking_init,
            step_size=new_step_size,
            current_y=state.current_y,
            current_f=state.current_f,
            best_f_val=best_f_val,
            diff=diff,
            cached_diff=state.cached_diff,
            grad=state.grad,
            result=result,
            step=state.step + 1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BacktrackingState[Y],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        """Terminate when the Armijo condition is satisfied.

        The Armijo condition approximates `f` around `current_y` with a linear function
        `f(new_y) = f(y + diff) ~ f(y) + g^T diff`
        where `g` is the gradient of `f` at `y`. In other words, `g^T diff` is
        the amount we'd expect `f` to decrease if it were linear.

        The Armijo condition is that `f(new_y)` must do better than a linear
        approximation proportional to the one above:
        ```
        f(new_y) < f(x) + self.backtrack_slope * g^T diff
        ```
        """
        del tags
        # This should maybe include a cauchy_like condition as well, especially
        # if we pass `step_size=0`
        predicted_reduction = tree_inner_prod(state.grad, state.diff)
        satisfies_armijo = (
            state.best_f_val
            <= state.current_f + self.backtrack_slope * predicted_reduction
        )
        finished = cast(
            Array,
            (state.result != RESULTS.successful)
            | (satisfies_armijo & (predicted_reduction <= 0)),
        )
        return finished, state.result

    def buffers(self, state: _BacktrackingState[Y]) -> tuple[()]:
        return ()


BacktrackingArmijo.__init__.__doc__ = """**Arguments:**

- `descent`: An [`optimistix.AbstractDescent`][] object, describing how to map from
    step-size (a scalar) to update (in y-space).
- `gauss_newton`: `True` if this is used for a least squares problem, `False`
    otherwise.
- `decrease_factor`: The rate at which to backtrack, i.e.
    `next_stepsize = backtrack_slope * current_stepsize`. Must be greater than 0.
- `backtrack_slope`: The slope of of the linear approximation to
    `f` that the backtracking algorithm must exceed to terminate. Larger
    means stricter termination criteria. Must be between 0 and 1.
- `backtracking_init`: The first `step_size` the backtracking algorithm will
    try. Must be greater than 0.
"""
