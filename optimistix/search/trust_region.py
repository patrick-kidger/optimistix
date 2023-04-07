import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..custom_types import sentinel
from ..line_search import AbstractGLS, AbstractTRModel
from ..linear_operator import AbstractLinearOperator
from ..solution import RESULTS
from .misc import init_derivatives


def _update_state(state):
    return (
        state.delta,
        state.descent_dir,
        state.f_y,
        state.f_new,
        state.model_y,
        state.model_new,
        state.finished,
    )


def _init_state(state):
    return (
        state.descent_dir,
        state.model_y,
        state.model_new,
    )


class TRState(eqx.Module):
    delta: float
    descent_dir: PyTree[ArrayLike]
    f_y: Float[ArrayLike, " "]
    f_new: Float[ArrayLike, " "]
    vector: PyTree[ArrayLike]
    operator: AbstractLinearOperator
    model_y: Float[ArrayLike, " "]
    model_new: Float[ArrayLike, " "]
    finished: bool
    aux: PyTree


class ClassicalTrustRegion(AbstractGLS):
    model: AbstractTRModel
    high_cutoff: float = 0.99
    low_cutoff: float = 1e-2
    high_constant: float = 3.5
    low_constant: float = 0.25
    needs_gradient = True
    needs_hessian = True
    #
    # These are not the textbook choices for the trust region line search
    # parameter values. This choice of default parameters comes from Gould
    # et al. "Sensitivity of trust region algorithms to their parameters," which
    # empirically tried a large grid of trust region constants on a number of test
    # problems. This set of constants had on average the fewest required iterations
    # to converge.
    #

    def init(self, problem, y, args, options):
        try:
            delta_0 = options["delta_0"]
        except KeyError:
            raise ValueError(
                "delta_0 must be passed for trust region methods via \
                options. Standard use is to set `delta_0` to the accepted \
                trust region size from last step."
            )
        if not isinstance(self.model, AbstractTRModel):
            raise ValueError(
                f"The model: {self.model} is not an instance of AbstractTRModel \
                (it does not have a __call__ method), which is necessary for trust \
                region methods."
            )

        f_y, vector, operator, aux = init_derivatives(
            self.model,
            problem,
            y,
            self.needs_gradient,
            self.needs_hessian,
            options,
            args,
        )

        state = TRState(
            delta_0,
            sentinel,
            f_y,
            f_y,
            vector,
            operator,
            sentinel,
            sentinel,
            finished=False,
            aux=aux,
        )

        try:
            model_y = options["model_y"]
        except KeyError:
            model_y = self.model(y, state)

        descent_dir = self.model.descent_dir(
            state.delta, problem, y, state, args, options
        )

        state = eqx.tree_at(_init_state, state, (descent_dir, model_y, model_y))

        return state

    def step(self, problem, y, args, options, state):
        f_new = problem.fn((ω(y) + state.delta * ω(state.descent_dir)).ω)
        model_new = self.model((ω(y) + state.delta * ω(state.descent_dir)).ω, state)

        model_y = state.model_new
        f_y = state.f_new

        tr_ratio = (f_y - f_new) / (state.model_y - state.model_new)

        finished = tr_ratio > self.low_cutoff
        delta = jnp.where(
            tr_ratio >= self.high_cutoff, state.delta * self.high_constant, state.delta
        )
        delta = jnp.where(
            tr_ratio <= self.low_cutoff, state.delta * self.low_constant, state.delta
        )

        def accept_direction(model, problem, y, args, options, state):
            return state.descent_dir

        def reject_direction(model, problem, y, args, options, state):
            return model.descent_dir(delta, problem, y, state, args, options)

        # if the tr algorithm has terminated, don't compute the next descent dir.
        # I am expecting the direction to be computed in the init instead.
        cond_args = (self.model, problem, y, args, options, state)
        descent_dir = lax.cond(finished, accept_direction, reject_direction, cond_args)

        # this often finishes after 1 step! This updates delta, the f_i, the model
        # evals, and sets `finished = True` to end the iterations.
        state = eqx.tree_at(
            _update_state,
            state,
            (delta, descent_dir, f_y, f_new, model_y, model_new, finished),
        )

        return state

    def terminate(self, problem, y, args, options, state):
        result = jnp.where(
            jnp.isfinite(state.delta), RESULTS.successful, RESULTS.divergence
        )
        return state.finished, result
