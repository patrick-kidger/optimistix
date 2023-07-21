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
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import AbstractLineSearchState, Aux, Fn, sentinel, Y
from .._descent import AbstractDescent, AbstractLineSearch
from .._misc import (
    is_linear,
    sum_squares,
    tree_dot,
    tree_where,
)
from .._solution import RESULTS


def _descent_no_results(descent, args, options, step_size):
    diff, results = descent(step_size, args, options)
    return diff


# We could technically support linear predicted reduction for Gauss-Newton
# and quasi-Newton methods as well, but this is likely to be more confusing
# than it's worth, so instead we use it only when `ClassicalTrustRegion` is
# called by a gradient-based method.
def _predict_linear_reduction(
    vector: PyTree[Array],
    diff: PyTree[Array],
    args: PyTree,
):
    """Compute the expected decrease in loss from taking the step `diff` in a
    gradient-based method.

    `predict_linear_reduction` approximates how much `fn` should decrease if we
    take the step `diff` by locally approximating `fn` with a linear function.
    ie. if `g` is the gradient of `fn` at `y`, then
    ```
    predicted_linear_reduction(diff) = g^T diff
    ```

    **Arguments**:

    - `diff`: the difference between `y_current` and `y_new`, also the output of a
        `descent`.
    - `vector`: the residual vector if `gauss_newton=True`, the gradient vector
        otherwise.
    - `operator`: the Jacobian if `gauss_newton=True`, `None` otherwise
    - `args`: a pytree of array arguments passed to `fn`.

    **Returns**:

    The expected decrease in loss from moving from `y` to `y + diff`.
    """
    del args
    return tree_dot(vector, diff)


def _predict_quadratic_reduction(
    gauss_newton: bool,
    operator: lx.AbstractLinearOperator,
    vector: PyTree[Array],
    diff: PyTree[Array],
    args: PyTree,
) -> Scalar:
    """Compute the expected decrease in loss from taking the step `diff` in
    quasi-Newton and Gauss-Newton models.

    `predict_quadratic_reduction` approximates how much `fn` should decrease if we
    take the step `diff` by locally approximating `f` with a quadratic function.
    ie. if `B` is the approximation to the Hessian coming from the
    quasi-Newton method at `y` and `g` the gradient of `fn` at `y`, then
    ```
    predicted_quadratic_reduction(diff) = g^T diff + 1/2 diff^T B diff
    ```

    **Arguments**:

    - `gauss_newton`: a bool indicating if the quasi-Newton method is a Gauss-Newton
        method. If it is, `vector` is the residual vector and `operator`
        is the Jacobian, and the computation differs slightly from the standard
        quasi-Newton case where `vector` is the gradient and `operator` is the
        approximate Hessian.
    - `diff`: the difference between `y_current` and `y_new`, also the output of a
        `descent`.
    - `vector`: the residual vector if `gauss_newton=True`, the gradient vector
        otherwise.
    - `operator`: the Jacobian if `gauss_newton=True`, the approximate hessian
        otherwise.
    - `args`: a pytree of array arguments passed to `fn`.

    **Returns**:

    The expected decrease in loss from moving from `y` to `y + diff`.
    """
    # In the Gauss-Newton setting we compute
    # `0.5 * [(Jp + r)^T (Jp + r) - r^T r]`
    # to the equation in the docstring
    # when `B = J^T J` and `g = J^T r`.
    if gauss_newton:
        rtr = sum_squares(vector)
        jacobian_term = sum_squares((ω(operator.mv(diff)) + ω(vector)).ω)
        reduction = 0.5 * (jacobian_term - rtr)
    else:
        operator_quadratic = 0.5 * tree_dot(diff, operator.mv(diff))
        steepest_descent = tree_dot(vector, diff)
        reduction = (operator_quadratic**ω + steepest_descent**ω).ω
    return reduction


class _TrustRegionState(AbstractLineSearchState, Generic[Y]):
    next_init: Scalar
    current_y: Y
    current_f: Scalar
    best_f_val: Scalar
    cached_diff: Y
    predict_reduction: Callable
    finished: Bool[Array, ""]
    result: RESULTS


#
# NOTE: we handle both the Gauss-Newton and quasi-Newton case identically
# in `__call__`.
# In the quasi-Newton setting, the step is `-B^† g` where`B^†` is the Moore-Penrose
# pseudoinverse of the quasi-Newton matrix `B`, and `g` is the gradient of the function
# to minimise.
# In the Gauss-Newton setting, the step is `-J^† r` where
# `J^†` is the pseudoinverse of the Jacobian, and `r` is the residual vector.
# Throughout, we have abstracted `J` and `B` into `operator` and
# `g` and `r` into `vector`, so the solves are identical.
#
# The gauss_newton flag is still necessary for the predicted reduction, whose
# computation does change depending on whether `operator` and `vector` contains the
# quasi-Newton matrix and vector or the Jacobian and residual. In line-searches which
# do not use this, there is no difference between `gauss_newton=True` and
# `gauss_newton=False`.
#


class _AbstractTrustRegion(AbstractLineSearch[_TrustRegionState[Y], Y, Aux]):
    """The abstract base class of the trust-region update algorithm.

    Trust region line searches compute the ratio
    `true_reduction/predicted_reduction`, where `true_reduction` is the decrease in `fn`
    between `y` and `new_y`, and `predicted_reduction` is how much we expected the
    function to decrease using a linear approximation to `fn`.

    The trust-region ratio determines whether to accept or reject a step and the
    next choice of step-size. Specifically:

    - reject the step and decrease stepsize if the ratio is smaller than a
        cutoff `low_cutoff`
    - accept the step and increase the step-size if the ratio is greater than
        another cutoff `high_cutoff` with `low_cutoff < high_cutoff`.
    - else, accept the step and make no change to the step-size.
    """

    descent: AbstractDescent
    gauss_newton: bool
    high_cutoff: float
    low_cutoff: float
    high_constant: float
    low_constant: float
    #
    # Note: we never actually compute the ratio
    # `true_reduction/predicted_reduction`. Instead, we rewrite the conditions as
    # `true_reduction < const * predicted_reduction` instead, where the inequality
    # flips because we assume `predicted_reduction` is negative.
    # This is for numerical reasons, it avoids an uneccessary subtraction and division.
    #
    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."
    #
    #
    # Technical note: when using a gradient-based method, `ClassicalTrustRegion` is a
    # variant of `BacktrackingLineSearch` with the Armijo condition, since the linear
    # predicted reduction is the same as the Armijo condition. However, unlike standard
    # backtracking, `ClassicalTrustRegion` chooses an initial backtracking length
    # depending on how well it did in the previous iteration.
    #

    def __post_init__(self):
        # You would not expect `self.low_cutoff` or `self.high_cutoff` to
        # be below zero, but this is technically not incorrect so we don't
        # require it.
        if self.low_cutoff > self.high_cutoff:
            raise ValueError(
                "`low_cutoff` must be below `high_cutoff` in `ClassicalTrustRegion`"
            )
        if self.low_constant < 0:
            raise ValueError(
                "`low_constant` must be greater than `0` in `ClassicalTrustRegion`"
            )
        if self.high_constant < 0:
            raise ValueError(
                "`high_constant` must be greater than `0` in `ClassicalTrustRegion`"
            )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _TrustRegionState,
        tags: frozenset[object],
    ) -> tuple[Y, _TrustRegionState[Y], Aux]:
        if state.cached_diff is sentinel:
            diff, result = self.descent(state.next_init, args, options)
        else:
            diff = (state.next_init * state.cached_diff**ω).ω
            result = state.result
        proposed_y = (state.current_y**ω + diff**ω).ω
        f_val, aux = fn(proposed_y, args)

        predicted_reduction = state.predict_reduction(diff, args)
        finished = f_val <= state.current_f + self.low_cutoff * predicted_reduction
        good = f_val < state.current_f + self.high_cutoff * predicted_reduction
        # If `predicted_reduction` is greater than 0, then it doesn't matter if we
        # beat it, we may still have gotten worse and need to reject the step.
        finished = finished & (predicted_reduction <= 0)
        good = good & (predicted_reduction < 0)

        new_min = f_val < state.best_f_val
        new_y = tree_where(new_min, proposed_y, y)
        best_f_val = jnp.where(new_min, f_val, state.best_f_val)
        new_step_size = jnp.where(
            good, state.next_init * self.high_constant, state.next_init
        )
        new_step_size = jnp.where(
            jnp.invert(finished), state.next_init * self.low_constant, new_step_size
        )
        new_step_size = jnp.where(
            new_step_size < jnp.finfo(new_step_size.dtype).eps,
            jnp.array(1.0),
            new_step_size,
        )
        new_state = _TrustRegionState(
            next_init=new_step_size,
            current_y=state.current_y,
            current_f=state.current_f,
            best_f_val=best_f_val,
            cached_diff=state.cached_diff,
            predict_reduction=state.predict_reduction,
            finished=finished,
            result=result,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _TrustRegionState[Y],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.finished, state.result

    def buffers(self, state: _TrustRegionState[Y]) -> tuple[()]:
        return ()


class ClassicalTrustRegion(_AbstractTrustRegion):
    """The classic trust-region update algorithm which uses a quadratic approximation of
    the objective function to predict reduction.

    This requires the following `options`:

    - `f0`: The value of the function to perform at line search at the point `y`.
    - `init_step_size`: The initial `step_size` that the line search will try.
    - `vector`: The residual vector if `gauss_newton=True`, the gradient vector
        otherwise.
    - `operator`: The Jacobian operator if `gauss_newton=True`, the approximate
        Hessian otherwise.
    """

    descent: AbstractDescent
    gauss_newton: bool
    high_cutoff: float = 0.99
    low_cutoff: float = 0.01
    high_constant: float = 3.5
    low_constant: float = 0.25

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _TrustRegionState[Y]:
        f0 = options["f0"]
        vector = options["vector"]
        step_size = options["init_step_size"]

        try:
            operator = options["operator"]
        except KeyError:
            raise ValueError(
                "`ClassicalTrustRegion` requires `operator` to be "
                "passed via `options`."
            )

        predicted_reduction = eqx.Partial(
            _predict_quadratic_reduction, self.gauss_newton, operator, vector
        )
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
        return _TrustRegionState(
            next_init=step_size,
            current_y=y,
            current_f=f0,
            best_f_val=f0,
            cached_diff=cached_diff,
            predict_reduction=predicted_reduction,
            finished=jnp.array(False),
            result=result,
        )


class LinearTrustRegion(_AbstractTrustRegion):
    """The trust-region update algorithm which uses a linear approximation of the
    objective function to predict reduction.

    This requires the following `options`:

    - `f0`: The value of the function to perform at line search at the point `y`.
    - `init_step_size`: The initial `step_size` that the line search will try.
    - `vector`: The residual vector if `gauss_newton=True`, the gradient vector
        otherwise.
    """

    descent: AbstractDescent
    gauss_newton: bool
    high_cutoff: float = 0.99
    low_cutoff: float = 0.01
    high_constant: float = 3.5
    low_constant: float = 0.25

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _TrustRegionState[Y]:
        f0 = options["f0"]
        vector = options["vector"]
        step_size = options["init_step_size"]
        predict_reduction = eqx.Partial(_predict_linear_reduction, vector)

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
        return _TrustRegionState(
            next_init=step_size,
            current_y=y,
            current_f=f0,
            best_f_val=f0,
            cached_diff=cached_diff,
            predict_reduction=predict_reduction,
            finished=jnp.array(False),
            result=result,
        )


_init_doc = """In the following, `ratio` refers to the ratio
`true_reduction/predicted_reduction`.

**Arguments**:

- `descent`: a `descent` object to compute what update to take given a step-size.
- `gauss_newton`: is backtracking a subroutine in least squares problem (True) or a
minimisation problem (False)?
- `high_cutoff`: the cutoff such that `ratio > high_cutoff` will accept the step
and increase the step-size on the next iteration.
- `low_cutoff`: the cutoff such that `ratio < low_cutoff` will reject the step
and decrease the step-size on the next iteration, and `ratio > low_cutoff`
will accept the step with no change to the step-size.
- `high_constant`: when `ratio > high_cutoff`, multiply the previous step-size by
high_constant`.
- `low_constant`: when `ratio < low_cutoff`, multiply the previous step-size by
low_constant`.
"""

LinearTrustRegion.__init__.__doc__ = _init_doc
ClassicalTrustRegion.__init__.__doc__ = _init_doc
