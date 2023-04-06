import abc
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from equinox import ω
from jaxtyping import ArrayLike, Float, PyTree

import optimistix as optx

from ..custom_types import sentinel
from ..line_search import AbstractGLS, AbstractModel
from ..solution import RESULTS
from .misc import init_derivatives


class BacktrackingState(eqx.Module):
    delta: float
    descent_dir: PyTree[ArrayLike]
    f_y: Float[ArrayLike, " "]
    f_new: Float[ArrayLike, " "]
    residual: PyTree[ArrayLike]
    model: AbstractModel
    gradient: PyTree[ArrayLike]
    hessian: PyTree[ArrayLike]
    slope: float
    decrease_factor: float
    aux: PyTree


def _update_state(state):
    return (state.delta, state.descent_dir)


def _update_state_fvals(state):
    return (state.f_y, state.f_new)


class AbstractBacktrackingGLS(AbstractGLS):
    backtrack_slope: float
    decrease_factor: float
    needs_gradient: ClassVar[bool]
    needs_hessian: ClassVar[bool]

    def init(self, problem, y, args, options):
        try:
            delta_0 = options["delta_0"]
        except KeyError:
            delta_0 = 1.0
        try:
            (f_y, aux) = options["f_and_aux"]
        except KeyError:
            if problem.has_aux:
                (f_y, aux) = problem.fn(y, args)
            else:
                f_y = problem.fn(y, args)
                aux = None
        gradient, hessian = init_derivatives(
            problem, y, self.needs_gradient, self.needs_hessian, options
        )
        state = BacktrackingState(
            delta_0,
            sentinel,
            f_y,
            f_y,
            gradient,
            hessian,
            self.backtrack_slope,
            self.decrease_factor,
            aux,
        )
        try:
            model = options["model"]
        except KeyError:
            raise ValueError("A model must be passed via the line-search options!")
        descent_dir = model.descent_dir(state.delta, state)
        state = eqx.tree_at(lambda x: x.descent_dir, state, descent_dir)
        return state

    def step(self, problem, y, args, options, state):
        delta = state.delta * state.decrease_factor
        f_y = state.f_new

        f_new = problem.fn((ω(state.y) + delta * ω(state.gradient)).ω)

        state = eqx.tree_at(_update_state_fvals, state, (f_y, f_new))
        descent_dir = state.model.descent_dir(delta, state)

        state = eqx.tree_at(_update_state, state, (delta, descent_dir))

        return state

    @abc.abstractmethod
    def terminate(self, problem, y, args, options, state):
        ...


class BacktrackingArmijo(optx.AbstractBacktrackingGLS):
    backtrack_slope: float = 0.1
    decrease_factor: float = 0.5
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = False

    def terminate(self, carry):
        state, delta = carry
        result = jnp.where(
            jnp.isfinite(state.delta), RESULTS.successful, RESULTS.divergence
        )
        linear_decrease = (ω(state.descent_dir.T) @ ω(state.gradient)).ω
        return state.f_new < state.f_y + state.slope * delta * linear_decrease, result
