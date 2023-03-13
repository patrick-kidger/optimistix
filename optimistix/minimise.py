from typing import Any, Dict, FrozenSet, Optional, TypeVar

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .solution import Solution


_SolverState = TypeVar("_SolverState")


class MinimiseProblem(AbstractIterativeProblem):
    tags: FrozenSet[object] = frozenset()


class AbstractMinimiser(AbstractIterativeSolver):
    pass


def _minimum(optimum, _, inputs, __):
    minimise_prob, args = inputs
    del inputs
    return jax.grad(minimise_prob.fn)(optimum, args)


@eqx.filter_jit
def minimise(
    problem: MinimiseProblem,
    solver: AbstractMinimiser,
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
        rewrite_fn=_minimum,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=throw,
        tags=problem.tags,
    )
