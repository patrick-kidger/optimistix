from typing import Callable, ClassVar

import jax.numpy as jnp

from ..line_search import AbstractModel
from ..linear_solve import linear_solve


class UnnormalizedGradient(AbstractModel):
    gauss_newton: ClassVar[bool] = False
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False

    def descent_dir(self, delta, state):
        return -delta * state.vector


class UnnormalizedNewton(AbstractModel):
    gauss_newton: bool
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False

    def descent_dir(self, delta, state):
        return -delta * linear_solve(state.operator, state.vector)


class NormalizedGradient(AbstractModel):
    gauss_newton: ClassVar[bool] = False
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False
    norm: Callable = jnp.linalg.norm

    def descent_dir(self, delta, state):
        return -delta * state.vector / self.norm(state.vector)


class NormalizedNewton(AbstractModel):
    gauss_newton: bool
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False
    norm: Callable = jnp.linalg.norm

    def descent_dir(self, delta, state):
        newton = linear_solve(state.operator, state.vector)
        return -delta * newton / self.norm(newton)
