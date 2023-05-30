from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Int, PyTree

from ..line_search import OneDimensionalFunction
from ..minimise import AbstractMinimiser, MinimiseProblem
from ..misc import tree_full, tree_where, tree_zeros_like, two_norm
from ..solution import RESULTS


class TRState(eqx.Module):
    f0: Array
    running_min: Array
    running_min_diff: PyTree[Array]
    finished: Array
    compute_f0: Bool[Array, " "]
    result: RESULTS
    step: Int[Array, ""]


def _get_predicted_reduction(options, diff):
    # This just exists for readibility, and because it may be
    # moved to solver/misc in the future
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
# NOTE: typically classical trust region methods compute
# (true decrease)/(predicted decrease) > const. We use
# -(true decrease) < const * -(predicted decrease) instead.
# This is for numerical reasons, as it avoids an uneccessary subtraction and division
#
class ClassicalTrustRegion(AbstractMinimiser):
    high_cutoff: float = 0.99
    low_cutoff: float = 0.01
    high_constant: float = 3.5
    low_constant: float = 0.25
    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."

    def first_init(self, vector, operator, options):
        # The natural init for quasi-Newton methods using trust regions
        # is to set the initial trust region to the full quasi-Newton step. This
        # turns the first use of the trust region algorithm into a standard
        # backtrackinga algortihm with a (possibly) nonlinear predicted reduction.
        # The user can pass `options["inti_line_search"]` to the overall solver
        # to set this explicitly.
        try:
            init_size = options["init_line_search"]
        except KeyError:
            # Do we want an API for passing the solver for this?
            newton = lx.linear_solve(
                operator, vector, lx.AutoLinearSolver(well_posed=False)
            ).value
            init_size = two_norm(newton)
        return init_size

    def init(
        self,
        problem: MinimiseProblem[OneDimensionalFunction],
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
    ):
        try:
            f0 = options["f0"]
            compute_f0 = options["compute_f0"]
        except KeyError:
            f0 = tree_full(f_struct, jnp.inf)
            compute_f0 = jnp.array(True)

        try:
            diff0 = tree_zeros_like(options["diff"])
        except KeyError:
            assert False

        state = TRState(
            f0=f0,
            running_min=f0,
            running_min_diff=diff0,
            finished=jnp.array(False),
            compute_f0=compute_f0,
            result=jnp.array(RESULTS.successful),
            step=jnp.array(0),
        )
        return state

    def step(
        self,
        problem: MinimiseProblem[OneDimensionalFunction],
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: TRState,
    ):
        y_or_zero = cast(Array, jnp.where(state.compute_f0, jnp.array(0.0), y))
        (f_new, (_, diff, aux, result, _)) = problem.fn(y_or_zero, args)
        predicted_reduction = _get_predicted_reduction(options, diff)
        # This is to make sure that `finished` and `good` are false on the first step.
        f0 = jnp.where(state.compute_f0, -jnp.inf, state.f0)
        finished = f_new < f0 + self.low_cutoff * predicted_reduction
        good = f_new < f0 + self.high_cutoff * predicted_reduction
        bad = f_new > f0 + self.low_cutoff * predicted_reduction
        # We don't want to change the size of the TR radius at first step.
        bad = bad & jnp.invert(state.compute_f0)
        # If `predicted_reduction` is greater than 0, then it doesn't matter if we
        # beat it, we may still have gotten worse and need to decrease the
        # trust region radius size.
        predicted_reduction_neg = predicted_reduction < 0
        finished = finished & predicted_reduction_neg
        good = good & predicted_reduction_neg
        bad = bad | jnp.invert(predicted_reduction_neg)
        new_y = jnp.where(good, y * self.high_constant, y)
        new_y = jnp.where(bad, y * self.low_constant, new_y)
        new_y = jnp.where(state.compute_f0, y, new_y)
        running_min = jnp.where(f_new < state.running_min, f_new, state.running_min)
        running_min_diff = tree_where(
            f_new < state.running_min, diff, state.running_min_diff
        )
        f0 = jnp.where(state.compute_f0, f_new, state.f0)
        new_state = TRState(
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
        problem: MinimiseProblem[OneDimensionalFunction],
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: TRState,
    ):
        result = jnp.where(
            jnp.isfinite(y),
            state.result,  # pyright: ignore
            RESULTS.nonlinear_divergence,  # pyright: ignore
        )
        return state.finished, result

    def buffers(self, state: TRState):
        return ()
