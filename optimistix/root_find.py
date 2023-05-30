from typing import Any, FrozenSet, Optional, TypeVar

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


def _root(root, _, inputs):
    root_prob, args, *_ = inputs
    del inputs

    def root_no_aux(x):
        if root_prob.has_aux:
            out, _ = root_prob.fn(x, args)
        else:
            out = root_prob.fn(x, args)
        return out

    return root_no_aux(root)


@eqx.filter_jit
def root_find(
    problem: RootFindProblem,
    solver: AbstractRootFinder,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[dict[str, Any]] = None,
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
