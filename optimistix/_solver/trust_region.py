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

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Fn, LineSearchAux, sentinel
from .._line_search import AbstractLineSearch
from .._misc import tree_full_like, tree_where, tree_zeros_like, two_norm
from .._solution import RESULTS


class _TrustRegionState(eqx.Module):
    f0: Array
    running_min: Array
    running_min_diff: PyTree[Array]
    finished: Array
    compute_f0: Bool[Array, " "]
    result: RESULTS
    step: Int[Array, ""]


def _get_predicted_reduction(options: dict[str, Any], diff: PyTree[Array]) -> Scalar:
    try:
        predicted_reduction_fn = options["predicted_reduction"]
        if predicted_reduction_fn is None:
            raise ValueError(
                "Expected a predicted reduction, got `None`. "
                "This is likely because a descent without a predicted reduction "
                "was passed."
            )
        else:
            predicted_reduction = predicted_reduction_fn(diff)
    except KeyError:
        raise ValueError(
            "The predicted reduction function must be passed to the "
            "classical trust region line search via `options['predicted_reduction']`"
        )
    return predicted_reduction


#
# Note that in `options` we anticipate that `f0`, `compute_f0`,
# `predicted_reduction` and 'diff' are passed. `compute_f0` indicates that
# this is the very first time that the line search has been called, and the search
# must compute `f(y)` --`y` the point at which the line search is initiated -- before
# continuing the search.
#
# Note that `aux` and `f_val` are the output of `f` at the END of the line search.
# ie. we don't return `f(y)`, but rather `f(y_new)` where `y_new` is the value
# returned by the line search.
# This is to exploit FSAL in our solvers, where the `f(y_new)` value is then
# passed as `f0` in the next line search.
#
# NOTE: typically trust region methods compute
# (true decrease)/(predicted decrease) > const. We use
# (true decrease) < const * (predicted decrease) instead (inequality flips because
# we assume predicted reduction is negative.)
# This is for numerical reasons, it avoids an uneccessary subtraction and division.
#
class ClassicalTrustRegion(AbstractLineSearch[_TrustRegionState]):
    high_cutoff: float = 0.99
    low_cutoff: float = 0.01
    high_constant: float = 3.5
    low_constant: float = 0.25
    init_linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    search_init: Scalar = sentinel
    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."

    def first_init(
        self,
        vector: PyTree[Array],
        operator: lx.AbstractLinearOperator,
        options: dict[str, Any],
    ) -> Scalar:
        if self.search_init == sentinel:
            # The natural init for quasi-Newton methods using trust regions
            # is to set the initial trust region to the full quasi-Newton step. This
            # turns the first use of the trust region algorithm into a standard
            # backtrackinga algortihm with a more general predicted reduction.
            newton = lx.linear_solve(operator, vector, self.init_linear_solver).value
            init_size = two_norm(newton)
        else:
            init_size = self.search_init

        return init_size

    def init(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _TrustRegionState:
        try:
            f0 = options["f0"]
            compute_f0 = options["compute_f0"]
        except KeyError:
            f0 = tree_full_like(f_struct, jnp.inf)
            compute_f0 = jnp.array(True)
        try:
            diff = tree_zeros_like(options["diff"])
        except KeyError:
            raise ValueError(
                "`diff` must be passed operator " "via `options['operator']`"
            )
        state = _TrustRegionState(
            f0=f0,
            running_min=f0,
            running_min_diff=diff,
            finished=jnp.array(False),
            compute_f0=compute_f0,
            result=RESULTS.successful,
            step=jnp.array(0),
        )
        return state

    def step(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _TrustRegionState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _TrustRegionState, LineSearchAux]:
        y_or_zero = jnp.where(state.compute_f0, jnp.array(0.0), y)
        (f_new, (_, diff, aux, result, _)) = fn(y_or_zero, args)

        predicted_reduction = _get_predicted_reduction(options, diff)
        finished = f_new < state.f0 + self.low_cutoff * predicted_reduction
        good = f_new < state.f0 + self.high_cutoff * predicted_reduction
        bad = f_new > state.f0 + self.low_cutoff * predicted_reduction
        # If `predicted_reduction` is greater than 0, then it doesn't matter if we
        # beat it, we may still have gotten worse and need to decrease the
        # trust region radius size.
        finished = finished & (predicted_reduction < 0) & jnp.invert(state.compute_f0)
        good = good & (predicted_reduction < 0)
        bad = bad | jnp.invert(predicted_reduction < 0)

        new_y = jnp.where(good, y * self.high_constant, y)
        new_y = jnp.where(bad, y * self.low_constant, new_y)
        running_min = jnp.where(f_new < state.running_min, f_new, state.running_min)
        running_min_diff = tree_where(
            f_new < state.running_min, diff, state.running_min_diff
        )
        # If `state.compute_f0` ignore the solution of the line search and set `f0`.
        new_y = jnp.where(state.compute_f0, y, new_y)
        f0 = jnp.where(state.compute_f0, f_new, state.f0)
        new_state = _TrustRegionState(
            f0=f0,
            running_min=running_min,
            running_min_diff=running_min_diff,
            finished=finished,
            compute_f0=jnp.array(False),
            result=result,
            step=state.step + 1,
        )
        return new_y, new_state, (running_min, running_min_diff, aux, result, new_y)

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _TrustRegionState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        result = RESULTS.where(
            jnp.isfinite(y),
            state.result,
            RESULTS.nonlinear_divergence,
        )
        return state.finished, result

    def buffers(self, state: _TrustRegionState) -> tuple[()]:
        return ()
