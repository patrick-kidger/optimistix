from typing import Any, Callable, Dict, Optional, TypeVar, Union

import equinox as eqx
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .linear_operator import Pattern
from .solution import Solution


_SolverState = TypeVar("_SolverState")


class RootFindProblem(AbstractIterativeProblem):
    pattern: Pattern


class AbstractRootFinder(AbstractIterativeSolver):
    pass


def _root(root, _, inputs, __):
    root_fn, args = inputs
    del inputs
    return root_fn(root, args)


@eqx.filter_jit
def root_find(
    root_fn: Union[Callable, RootFindProblem],
    solver: AbstractRootFinder,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
) -> Solution:
    if isinstance(root_fn, RootFindProblem):
        root_prob = root_fn
    else:
        root_prob = RootFindProblem(root_prob, has_aux=False, pattern=Pattern())
    del root_fn

    return iterative_solve(
        root_prob,
        solver,
        y0,
        args,
        options,
        rewrite_fn=_root,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=throw,
        pattern=root_prob.pattern,
    )
