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


class FixedPointProblem(AbstractIterativeProblem):
    pass


class AbstractFixedPointSolver(AbstractIterativeSolver):
    pass


def _fixed_point(root, _, inputs, __):
    fixed_point_prob, args = inputs
    del inputs
    return fixed_point_prob.fn(root, args) - root


class _ToRootFn(eqx.Module):
    fixed_point_fn: Callable

    def __call__(self, y, args):
        return self.fixed_point_fn(y, args) - y


@eqx.filter_jit(donate="none")
def fixed_point(
    fixed_point_fn: Union[Callable, FixedPointProblem],
    solver: Union[AbstractFixedPointSolver, AbstractRootFinder],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
) -> Solution:
    if isinstance(fixed_point_fn, FixedPointProblem):
        fixed_point_prob = fixed_point_fn
    else:
        fixed_point_prob = FixedPointProblem(fn=fixed_point_fn, has_aux=False)
    del fixed_point_fn

    if jax.eval_shape(lambda: y0) != jax.eval_shape(fixed_point_prob.fn, y0, args):
        raise ValueError(
            "The input and output of `fixed_point_fn` must have the same structure"
        )

    if isinstance(solver, AbstractRootFinder):
        root_fn = _ToRootFn(fixed_point_prob.fn)
        root_prob = RootFindProblem(
            fn=root_fn, has_aux=fixed_point_prob.has_aux, pattern=Pattern()
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
            fixed_point_prob,
            solver,
            y0,
            args,
            options,
            rewrite_fn=_fixed_point,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
