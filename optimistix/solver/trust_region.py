import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike, Bool, Float

from ..line_search import AbstractLineSearch
from ..solution import RESULTS


class TRState(eqx.Module):
    f_prev: Float[ArrayLike, " "]
    finished: Bool[ArrayLike, " "]


#
# NOTE: typically classical trust region methods compute
# (true decrease)/(predicted decrease) > const. We use
# -(true decrease) < const * -(predicted decrease) instead.
# This is for numerical reasons, as it avoids an uneccessary subtraction and division
#
class ClassicalTrustRegion(AbstractLineSearch):
    high_cutoff: float = 0.99
    low_cutoff: float = 0.01
    high_constant: float = 3.5
    low_constant: float = 0.25
    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."

    def init(self, problem, y, args, options):
        # NOTE: passing `f_0` via options to exploit FSAL. We could also add default
        # behavior by computing `problem.fn(jnp.array(0.), args)` at the cost of
        # compilation time.
        try:
            f_0 = options["f_0"]
        except KeyError:
            raise ValueError("f_0 must be passed via options to classical trust region")

        state = TRState(
            f_0,
            jnp.array(False),
        )

        return state

    def step(self, problem, y, args, options, state):
        (f_new, (descent_dir, aux)) = problem.fn(y, args)
        # WARNING: this `problem.fn.descent_fn.fn` is annoying. However, I don't have a
        # great way out of this.
        try:
            vector = options["vector"]
            operator = options["operator"]
            predicted_reduction = problem.fn.descent_fn.fn.predicted_reduction(
                descent_dir, args, state, options, vector, operator
            )
        except KeyError:
            try:
                predicted_reduction_fn = options["predicted_reduction"]
                predicted_reduction = predicted_reduction_fn(
                    descent_dir, args, state, options
                )
            except KeyError:
                raise ValueError(
                    "Need a method to predict reduction. \
                    This can be achieved by passing `vector` and `operator` via \
                    options, or by passing `predicted_reduction_fn` via options."
                )

        finished = f_new < state.f_prev + self.low_cutoff * predicted_reduction
        good = f_new < state.f_prev + self.high_cutoff * predicted_reduction
        bad = f_new > state.f_prev + self.low_cutoff * predicted_reduction

        new_y = jnp.where(good, y * self.high_constant, y)

        new_y = jnp.where(bad, y * self.low_constant, new_y)

        new_state = TRState(f_new, finished)

        return new_y, new_state, (descent_dir, aux)

    def terminate(self, problem, y, args, options, state):
        result = jnp.where(
            jnp.isfinite(y), RESULTS.successful, RESULTS.nonlinear_divergence
        )
        return (state.finished, result)

    def buffer(self, state):
        return ()
