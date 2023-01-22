from typing import Any, Callable, Dict, Optional, TypeVar, Union

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .linear_operator import Pattern
from .root_find import AbstractRootFinder, root_find, RootFindProblem
from .solution import Solution


_SolverState = TypeVar("_SolverState")


class MinimiseProblem(AbstractIterativeProblem):
    pass


class AbstractMinimiser(AbstractIterativeSolver):
    pass


def _minimum(optimum, _, inputs, __):
    minimise_prob, args = inputs
    del inputs
    return jax.grad(minimise_prob.fn)(optimum, args)


class _ToRootFn(eqx.Module):
    minimise_fn: Callable

    def __call__(self, y, args):
        return jax.grad(self.minimise_fn)(y, args)


@eqx.filter_jit(donate="none")
def minimise(
    minimise_fn: Union[Callable, MinimiseProblem],
    solver: Union[AbstractMinimiser, AbstractRootFinder],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
) -> Solution:
    if isinstance(minimise_fn, MinimiseProblem):
        minimise_prob = minimise_fn
    else:
        minimise_prob = MinimiseProblem(minimise_fn, has_aux=False)
    del minimise_fn

    if isinstance(solver, AbstractRootFinder):
        root_fn = _ToRootFn(minimise_prob.fn)
        root_prob = RootFindProblem(
            root_fn, has_aux=minimise_prob.has_aux, pattern=Pattern(symmetric=True)
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
    else:
        return iterative_solve(
            minimise_prob,
            solver,
            y0,
            args,
            options,
            rewrite_fn=_minimum,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
