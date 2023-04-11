from typing import Callable, ClassVar

import jax.numpy as jnp
from equinox.internal import ω

from ..line_search import AbstractModel
from ..linear_solve import linear_solve


class UnnormalizedGradient(AbstractModel):
    gauss_newton: ClassVar[bool] = False
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = True

    def descent_dir(self, delta, problem, y, state, args, options):
        return -delta * state.vector


class UnnormalizedNewton(AbstractModel):
    gauss_newton: bool
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = True

    def descent_dir(self, delta, problem, y, state, args, options):
        return (-delta * ω(linear_solve(state.operator, state.vector).value)).ω


class NormalizedGradient(AbstractModel):
    gauss_newton: ClassVar[bool] = False
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = True
    norm: Callable = jnp.linalg.norm

    def descent_dir(self, delta, problem, y, state, args, options):
        return -delta * state.vector / self.norm(state.vector)


class NormalizedNewton(AbstractModel):
    gauss_newton: bool
    computes_operator: ClassVar[bool] = False
    computes_vector: ClassVar[bool] = False
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = True
    norm: Callable = jnp.linalg.norm

    def descent_dir(self, delta, problem, y, state, args, options):
        newton = linear_solve(state.operator, state.vector).value
        return (-delta * ω(newton) / self.norm(newton)).ω
