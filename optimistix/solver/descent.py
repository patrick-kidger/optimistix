from typing import ClassVar

import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from ..line_search import AbstractDescent
from ..linear_solve import AutoLinearSolver, linear_solve


class UnnormalizedGradient(AbstractDescent):
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = False

    def __call__(
        self, delta, delta_args, problem, y, args, state, options, vector, operator
    ):
        descent_dir = (-delta * ω(vector)).ω
        return descent_dir


class UnnormalizedNewton(AbstractDescent):
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = True

    def __call__(
        self, delta, delta_args, problem, y, args, state, options, vector, operator
    ):
        descent_dir = (
            -delta
            * ω(
                linear_solve(operator, vector, AutoLinearSolver(well_posed=False)).value
            )
        ).ω
        return descent_dir


class NormalizedGradient(AbstractDescent):
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = False

    def __call__(self, delta, problem, y, args, state, options, vector, operator):
        _sumsqr = lambda x: jnp.sum(x**2)
        vec_norm = jtu.tree_reduce(lambda x, y: x + y, ω(vector).call(_sumsqr).ω)
        descent_dir = (-delta * ω(vector) / vec_norm).ω
        return descent_dir


class NormalizedNewton(AbstractDescent):
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = False

    def __call__(
        self, delta, delta_args, problem, y, args, state, options, vector, operator
    ):
        newton = linear_solve(
            operator, vector, solver=AutoLinearSolver(well_posed=False)
        ).value
        _sumsqr = lambda x: jnp.sum(x**2)
        newton_norm = jtu.tree_reduce(lambda x, y: x + y, ω(newton).call(_sumsqr).ω)
        descent_dir = ((-delta * ω(newton)).ω / newton_norm).ω
        return descent_dir
