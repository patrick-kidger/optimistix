import equinox as eqx
import jax.numpy as jnp
from equinox import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..custom_types import sentinel
from ..line_search import AbstractGLS, AbstractTRModel
from ..solution import RESULTS
from .misc import init_derivatives


def _update_state(state):
    return (
        state.delta,
        state.f_y,
        state.f_new,
        state.model_y,
        state.model_new,
        state.finished,
    )


class TRState(eqx.Module):
    delta: float
    descent_dir: PyTree[ArrayLike]
    model: AbstractTRModel
    f_y: Float[ArrayLike, " "]
    f_new: Float[ArrayLike, " "]
    model_y: Float[ArrayLike, " "]
    model_new: Float[ArrayLike, " "]
    finished: bool
    aux: PyTree


class TrustRegionDecrease(AbstractGLS):
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
                options. Standard use is `delta_0` is accepted delta of trust \
                region last step."
            )
        try:
            (f_y, aux) = options["f_and_aux"]
        except KeyError:
            if problem.has_aux:
                (f_y, aux) = problem.fn(y, args)
            else:
                f_y = problem.fn(y, args)
                aux = None
        try:
            model = options["model"]
        except KeyError:
            raise ValueError("A model must be passed via the line-search options!")
        try:
            model_y = options["model_y"]
        except KeyError:
            model_y = model(y)

        if not isinstance(model, AbstractTRModel):
            raise ValueError(
                f"The model: {model} is not an instance of AbstractTRModel \
                (it does not have a __call__ method), which is necessary for trust \
                region methods."
            )

        gradient, hessian = init_derivatives(
            problem, y, self.needs_gradient, self.needs_hessian, options
        )
        state = TRState(
            delta_0,
            sentinel,
            model,
            f_y,
            f_y,
            model_y,
            model_y,
            finished=False,
            aux=aux,
        )
        descent_dir = model.descent_dir(state.delta, state)
        state = eqx.tree_at(lambda x: x.descent_dir, state, descent_dir)
        return state

    def step(self, problem, y, args, options, state):
        f_y = state.f_new
        f_new = problem.fn((ω(y) + state.delta * ω(state.descent_dir)))
        model_y = state.model_new
        model_new = state.model((ω(y) + state.delta * ω(state.descent_dir)))

        tr_ratio = (f_y - f_new) / (model_y - model_new)

        finished = tr_ratio > self.low_cutoff
        delta = jnp.where(
            tr_ratio >= self.high_cutoff, state.delta * self.high_constant, state.delta
        )
        delta = jnp.where(
            tr_ratio <= self.low_cutoff, state.delta * self.low_constant, state.delta
        )

        # this finishes after 1 step! this updates delta, the f_i, the model evals,
        # and sets `finished = True` to end the iterations.
        state = eqx.tree_at(
            _update_state, state, (delta, f_y, f_new, model_y, model_new, finished)
        )

        return state

    def terminate(self, problem, y, args, options, state):
        result = jnp.where(
            jnp.isfinite(state.delta), RESULTS.successful, RESULTS.divergence
        )
        return state.finished, result
