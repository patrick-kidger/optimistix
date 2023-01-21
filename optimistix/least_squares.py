from typing import Any, Callable, Dict, Optional, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .linear_operator import Pattern
from .minimise import AbstractMinimiser, minimise, MinimiseProblem
from .root_find import AbstractRootFinder, root_find, RootFindProblem
from .solution import Solution


_SolverState = TypeVar("_SolverState")


class LeastSquaresProblem(AbstractIterativeProblem):
    pattern: Pattern


class AbstractLeastSquaresSolver(AbstractIterativeSolver):
    pass


def _residual(optimum, _, inputs, __):
    residual_prob, args = inputs
    del inputs

    def objective(_optimum):
        return jnp.sum(residual_prob.fn(_optimum, args) ** 2)

    return jax.grad(objective)(optimum)


class _ToRootFn(eqx.Module):
    residual_fn: Callable

    def __call__(self, y, args):
        def objective(_y):
            return jnp.sum(self.residual_fn(_y, args) ** 2)

        return jax.grad(objective)(y)


class _ToMinimiseFn(eqx.Module):
    residual_fn: Callable

    def __call__(self, y, args):
        return jnp.sum(self.residual_fn(y, args) ** 2)


@eqx.filter_jit
def least_squares(
    residual_fn: Union[Callable, LeastSquaresProblem],
    solver: Union[AbstractLeastSquaresSolver, AbstractMinimiser, AbstractRootFinder],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
) -> Solution:
    if isinstance(residual_fn, LeastSquaresProblem):
        residual_prob = residual_fn
    else:
        residual_prob = LeastSquaresProblem(
            residual_fn, has_aux=False, pattern=Pattern()
        )
    del residual_fn

    if isinstance(solver, AbstractRootFinder):
        root_fn = _ToRootFn(residual_prob.fn)
        root_prob = RootFindProblem(
            fn=root_fn, has_aux=residual_prob.has_aux, pattern=Pattern(symmetric=True)
        )
        return root_find(
            root_prob,
            solver,
            y0,
            args,
            options,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )

    elif isinstance(solver, AbstractMinimiser):
        minimise_fn = _ToMinimiseFn(residual_prob.fn)
        minimise_prob = MinimiseProblem(fn=minimise_fn, has_aux=residual_prob.has_aux)
        return minimise(
            minimise_prob,
            solver,
            y0,
            args,
            options,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )

    else:
        return iterative_solve(
            residual_prob,
            solver,
            y0,
            args,
            options,
            rewrite_fn=_residual,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
