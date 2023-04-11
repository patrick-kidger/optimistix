import abc
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..custom_types import sentinel
from ..line_search import AbstractGLS, AbstractModel
from ..linear_operator import AbstractLinearOperator
from ..solution import RESULTS
from .misc import init_derivatives


class BacktrackingState(eqx.Module):
    delta: Float[ArrayLike, " "]
    descent_dir: PyTree[ArrayLike] | object
    f_y: Float[ArrayLike, " "]
    f_new: Float[ArrayLike, " "]
    operator: AbstractLinearOperator | object
    vector: PyTree[ArrayLike] | object
    aux: PyTree


def _update_state(state):
    return (state.delta, state.descent_dir, state.f_y, state.f_new)


class AbstractBacktrackingGLS(AbstractGLS):
    backtrack_slope: Float[ArrayLike, " "]
    decrease_factor: Float[ArrayLike, " "]
    needs_gradient: bool
    needs_hessian: bool

    def __call__(self, model):
        return _AbstractBacktrackingGLS(
            model,
            self.backtrack_slope,
            self.decrease_factor,
            self.needs_gradient,
            self.needs_hessian,
        )


class _AbstractBacktrackingGLS(AbstractGLS):
    model: AbstractModel
    backtrack_slope: Float[ArrayLike, " "]
    decrease_factor: Float[ArrayLike, " "]
    needs_gradient: ClassVar[bool]
    needs_hessian: ClassVar[bool]

    def init(self, problem, y, args, options):
        try:
            delta_0 = options["delta_0"]
        except KeyError:
            delta_0 = jnp.array(1.0)

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

        f_new = problem.fn((ω(y) + ω(descent_dir)).ω, args)
        if problem.has_aux:
            (f_new, aux) = f_new
        else:
            aux = None

        state = eqx.tree_at(_update_state, state, (delta, descent_dir, f_y, f_new))

        return delta, state, aux

    @abc.abstractmethod
    def terminate(self, problem, y, args, options, state):
        ...


class BacktrackingArmijo(AbstractBacktrackingGLS):
    backtrack_slope: float = 0.1
    decrease_factor: float = 0.5
    needs_gradient: bool = True
    needs_hessian: bool = False

    def __call__(self, model):
        return _BacktrackingArmijo(
            model=model,
            backtrack_slope=self.backtrack_slope,
            decrease_factor=self.decrease_factor,
            needs_gradient=self.needs_gradient,
            needs_hessian=self.needs_hessian,
        )


class _BacktrackingArmijo(_AbstractBacktrackingGLS):
    model: AbstractModel
    backtrack_slope: Float[ArrayLike, " "]
    decrease_factor: Float[ArrayLike, " "]
    needs_gradient: bool
    needs_hessian: bool

    def __post_init__(self):
        if self.needs_gradient or self.model.needs_gradient:
            self.needs_gradient = True
        if self.needs_hessian or self.model.needs_hessian:
            self.needs_hessian = True

    def terminate(self, problem, y, args, options, state):
        result = jnp.where(
            jnp.isfinite(state.delta), RESULTS.successful, RESULTS.nonlinear_divergence
        )

        if self.model.gauss_newton:
            gradient = state.operator.mv(state.vector)
        else:
            gradient = state.vector

        sum_leaves = (ω(state.descent_dir) * ω(gradient)).call(jnp.sum).ω
        linear_decrease = sum(jtu.tree_leaves(sum_leaves))

        return (
            state.f_new
            < state.f_y + self.backtrack_slope * state.delta * linear_decrease,
            result,
        )

    def buffer(self, state):
        return
