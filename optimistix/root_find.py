from typing import Any, Dict, FrozenSet, Optional, TypeVar

import equinox as eqx
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .solution import Solution


_SolverState = TypeVar("_SolverState")


class RootFindProblem(AbstractIterativeProblem):
    tags: FrozenSet[object] = frozenset()


class AbstractRootFinder(AbstractIterativeSolver):
    pass


def _root(root, _, inputs, __):
    root_fn, args = inputs
    del inputs
    return root_fn(root, args)


@eqx.filter_jit
def root_find(
    problem: RootFindProblem,
    solver: AbstractRootFinder,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
) -> Solution:
    return iterative_solve(
        problem,
        solver,
        y0,
        args,
        options,
        rewrite_fn=_root,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=throw,
        tags=problem.tags,
    )
