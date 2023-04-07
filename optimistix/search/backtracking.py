import abc
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..custom_types import sentinel
from ..line_search import AbstractGLS, AbstractModel
from ..linear_operator import AbstractLinearOperator
from ..solution import RESULTS
from .misc import init_derivatives


class BacktrackingState(eqx.Module):
    delta: float
    descent_dir: PyTree[ArrayLike]
    f_y: Float[ArrayLike, " "]
    f_new: Float[ArrayLike, " "]
    operator: AbstractLinearOperator
    vector: PyTree[ArrayLike]
    aux: PyTree


def _update_state(state):
    return (state.delta, state.descent_dir, state.f_y, state.f_new)


class AbstractBacktrackingGLS(AbstractGLS):
    model: AbstractModel
    backtrack_slope: float
    decrease_factor: float
    needs_gradient: ClassVar[bool]
    needs_hessian: ClassVar[bool]

    def init(self, problem, y, args, options):
        try:
            delta_0 = options["delta_0"]
        except KeyError:
            delta_0 = 1.0

        # see the definition of init_derivatives for an explanation of this terminology
        f_y, vector, operator, aux = init_derivatives(
            self.model,
            problem,
            y,
            self.needs_gradient,
            self.needs_hessian,
            options,
            args,
        )
        state = BacktrackingState(
            delta=delta_0,
            descent_dir=sentinel,
            f_y=f_y,
            f_new=f_y,
            operator=operator,
            vector=vector,
            aux=aux,
        )
        descent_dir = self.model.descent_dir(delta_0, problem, y, state, args, options)
        state = eqx.tree_at(lambda x: x.descent_dir, state, descent_dir)
        return state

    def step(self, problem, y, args, options, state):
        delta = state.delta * self.decrease_factor
        f_y = state.f_new

        descent_dir = self.model.descent_dir(delta, problem, y, state, args, options)

        f_new = problem.fn((ω(y) + ω(descent_dir)).ω)

        state = eqx.tree_at(_update_state, state, (delta, descent_dir, f_y, f_new))

        return state

    @abc.abstractmethod
    def terminate(self, problem, y, args, options, state):
        ...


class BacktrackingArmijo(AbstractBacktrackingGLS):
    model: AbstractModel
    backtrack_slope: float = 0.1
    decrease_factor: float = 0.5
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = False

    def terminate(self, problem, y, args, options, state):
        result = jnp.where(
            jnp.isfinite(state.delta), RESULTS.successful, RESULTS.divergence
        )

        if self.model.gauss_newton:
            gradient = state.operator.mv(state.vector)
        else:
            gradient = state.vector

        linear_decrease = (ω(state.descent_dir.T) @ ω(gradient)).ω
        return (
            state.f_new
            < state.f_y + self.backtrack_slope * state.delta * linear_decrease,
            result,
        )
