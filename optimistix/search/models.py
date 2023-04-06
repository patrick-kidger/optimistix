from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import ArrayLike

import optimistix as optx

from ..custom_types import sentinel
from ..line_search import AbstractModel


class UnnormalizedGradient(AbstractModel):
    def descent_dir(self, delta, state):
        return -delta * state.gradient


class UnnormalizedNewton(AbstractModel):
    def descent_dir(self, delta, state):
        return -delta * optx.linear_solve(state.hessian, state.gradient)


class NormalizedGradient(AbstractModel):
    norm: Callable = jnp.linalg.norm

    def descent_dir(self, delta, state):
        return -delta * state.gradient / self.norm(state.gradient)


class NormalizedNewton(AbstractModel):
    norm: Callable = jnp.linalg.norm

    def descent_dir(self, delta, state):
        newton = optx.linear_solve(state.hessian, state.gradient)
        return -delta * newton / self.norm(newton)


class Dogleg(AbstractModel):
    tr_matrix: Optional[ArrayLike] = sentinel

    def descent_dir(self, delta, state):
        ...
