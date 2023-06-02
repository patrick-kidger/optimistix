from typing import Any, Callable, Optional, TypeVar, Union

import equinox as eqx
import jax
from equinox.internal import ω
from jaxtyping import Array, PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from ._root_find import AbstractRootFinder, root_find, RootFindProblem
from ._solution import Solution


_SolverState = TypeVar("_SolverState")


class FixedPointProblem(AbstractIterativeProblem):
    pass


class AbstractFixedPointSolver(AbstractIterativeSolver):
    pass


def _fixed_point(root, _, inputs):
    fixed_point_prob, args, *_ = inputs
    del inputs

    def fixed_point_no_aux(x):
        if fixed_point_prob.has_aux:
            out, _ = fixed_point_prob.fn(x, args)
        else:
            out = fixed_point_prob.fn(x, args)
        return out

    return (fixed_point_no_aux(root) ** ω - root**ω).ω


class _ToRootFn(eqx.Module):
    fixed_point_fn: Callable
    has_aux: bool

    def __call__(self, y, args):
        out = self.fixed_point_fn(y, args)
        if self.has_aux:
            out, aux = out
            return out - y, aux
        else:
            return out - y


@eqx.filter_jit
def fixed_point(
    problem: FixedPointProblem,
    solver: Union[AbstractFixedPointSolver, AbstractRootFinder],
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[dict[str, Any]] = None,
    *,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
) -> Solution:
    struct = jax.eval_shape(lambda: problem.fn(y0, args))
    if problem.has_aux:
        struct, aux_struct = struct
    else:
        aux_struct = None
    if jax.eval_shape(lambda: y0) != struct:
        raise ValueError(
            "The input and output of `fixed_point_fn` must have the same structure"
        )

    if isinstance(solver, AbstractRootFinder):
        root_fn = _ToRootFn(problem.fn, problem.has_aux)
        root_problem = RootFindProblem(
            fn=root_fn, has_aux=problem.has_aux, tags=frozenset()
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
            rewrite_fn=_fixed_point,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=frozenset(),
            aux_struct=aux_struct,
            f_struct=struct,
        )
