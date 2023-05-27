from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PyTree

from ..line_search import OneDimensionalFunction
from ..minimise import AbstractMinimiser, MinimiseProblem
from ..misc import tree_full
from ..solution import RESULTS


class TRState(eqx.Module):
    f_val: Array
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
        # in a sense, there is a natural scaling for quasi-Newton methods
        # where you'd normally apply a trust-region method. However, we don't
        # anticipate a user is always passing in these methods, or doesn't do
        # normalisation within the descent itself. As such, we choose a sub-optimal
        # default value `1`, and allow the user the pass this via options to the
        # external solver.
        try:
            init_size = options["init_line_search"]
        except KeyError:
            init_size = jnp.array(1.0)
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

        state = TRState(
            f_val=f0,
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
        (f_new, (_, diff, aux, result, _)) = problem.fn(y, args)
        # Q: Can I just pass this via state?
        predicted_reduction = _get_predicted_reduction(options, diff)
        # predicted_reduction should be < 0, this is a safety measure.
        f_prev = jnp.where(state.compute_f0, -jnp.inf, state.f_val)
        # TODO(raderj): switch this into a predicted_reduction < 0
        # check downstream, and add an indication that the line_search
        # failed and should not be taken into account in the termination
        # condition of the final solver
        # predicted_reduction = jnp.minimum(predicted_reduction, 0)
        # usually this is written in terms of a "trust region ratio", but this
        # is equivalent and slightly more numerically stable. Note that when
        # the predicted reduction is linear, `good` and `finished` are Armijo
        # conditions.
        finished = f_new < f_prev + self.low_cutoff * predicted_reduction
        finished = cast(
            Bool[Array, ""], jnp.where(state.compute_f0, jnp.array(True), finished)
        )
        finished = finished | state.compute_f0
        good = f_new < f_prev + self.high_cutoff * predicted_reduction
        good = good & jnp.invert(state.compute_f0)
        bad = f_new > f_prev + self.low_cutoff * predicted_reduction
        bad = bad & jnp.invert(state.compute_f0)

        predicted_reduction_neg = predicted_reduction < 0
        finished = finished & predicted_reduction_neg
        good = good & predicted_reduction_neg
        bad = bad | jnp.invert(predicted_reduction_neg)

        new_y = jnp.where(good, y * self.high_constant, y)
        new_y = jnp.where(bad, y * self.low_constant, new_y)
        f_new = jnp.where(finished | state.compute_f0, f_new, f_prev)
        new_state = TRState(
            f_val=f_new,
            finished=finished,
            compute_f0=jnp.array(False),
            result=result,
            step=state.step + 1,
        )
        return new_y, new_state, (f_new, diff, aux, result, new_y)

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
