import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..line_search import AbstractLineSearch
from ..solution import RESULTS


class BacktrackingState(eqx.Module):
    f_delta: Float[ArrayLike, " "]
    f_0: Float[ArrayLike, " "]
    descent_dir: PyTree[ArrayLike]


def _update_state(state):
    return (state.f_delta, state.descent_dir)


class AbstractBacktrackingLineSearch(AbstractLineSearch):
    backtrack_slope: Float[ArrayLike, " "]
    decrease_factor: Float[ArrayLike, " "]

    def init(self, problem, y, args, options):
        # TODO(raderj): remove this. Difficult only because
        # descent dir should be the right shape but it is difficult
        # to get something with the shape of descent dir
        (f_0, (descent_dir, _)) = problem.fn(0.0, args)
        state = BacktrackingState(f_delta=f_0, f_0=f_0, descent_dir=descent_dir)
        return state

    def step(self, problem, y, args, options, state):
        delta = y * self.decrease_factor

        (f_delta, (descent_dir, aux)) = problem.fn(y, args)

        state = eqx.tree_at(_update_state, state, (f_delta, descent_dir))

        return delta, state, (descent_dir, aux)

    @abc.abstractmethod
    def terminate(self, problem, y, args, options, state):
        ...


class BacktrackingArmijo(AbstractBacktrackingLineSearch):
    backtrack_slope: Float[ArrayLike, " "]
    decrease_factor: Float[ArrayLike, " "]

    def terminate(self, problem, y, args, options, state):
        result = jnp.where(
            jnp.isfinite(y), RESULTS.successful, RESULTS.nonlinear_divergence
        )

        # normally this will be presented in textbooks as g^T d where g
        # is the gradient of f and d is the descent direction. We assume that
        # problem.fn is the 1d line search, and so these are equivalent.
        try:
            vector = options["vector"]
            gradient = jtu.tree_reduce(
                lambda x, y: x + y, (ω(vector) * ω(state.descent_dir)).ω
            )
        except KeyError:
            (gradient, _) = jax.jacfwd(problem.fn)(0.0, args)

        return (
            state.f_delta < state.f_0 + self.backtrack_slope * y * gradient,
            result,
        )

    def buffer(self, state):
        return ()
