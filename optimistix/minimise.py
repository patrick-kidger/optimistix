from typing import Any, Callable, Dict, Optional, TypeVar, Union

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .root_find import AbstractRootFinder, root_find, RootFindProblem
from .linear_tags import symmetric_tag
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
    hax_aux: bool

    def __call__(self, y, args):
        return jax.grad(self.minimise_fn, has_aux=self.has_aux)(y, args)


@eqx.filter_jit
def minimise(
    problem: MinimiseProblem,
    solver: Union[AbstractMinimiser, AbstractRootFinder],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
) -> Solution:
    if isinstance(solver, AbstractRootFinder):
        root_fn = _ToRootFn(problem.fn, problem.has_aux)
        root_problem = RootFindProblem(
            root_fn, has_aux=problem.has_aux, tags=frozenset([symmetric_tag])
        )
        return root_find(
            root_problem,
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
            problem,
            solver,
            y0,
            args,
            options,
            rewrite_fn=_minimum,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
