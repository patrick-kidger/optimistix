from typing import Any, Callable, Dict, FrozenSet, Optional, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .misc import tree_where
from .solution import Solution


_SolverState = TypeVar("_SolverState")

MinimiseFunction = TypeVar("MinimiseFunction")


class MinimiseProblem(AbstractIterativeProblem[MinimiseFunction]):
    tags: FrozenSet[object] = frozenset()


class AbstractMinimiser(AbstractIterativeSolver):
    pass


class _ProblemPipeThrough(MinimiseProblem):
    def __init__(self, fn: Callable, has_aux: bool, tags: FrozenSet[object]):
        def auxmented(x, args):
            if has_aux:
                y, original_aux = fn(x, args)
                aux = (y, original_aux)
            else:
                y = fn(x, args)
                aux = y
            return y, aux

        self.fn = auxmented
        self.has_aux = True
        self.tags = tags


class RunningMinMinimiser(AbstractMinimiser):
    minimiser: AbstractMinimiser

    def init(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: PyTree,
        options: Dict[str, Any],
    ):
        f0, _ = jtu.tree_map(
            lambda x: jnp.full(x.shape, jnp.inf), jax.eval_shape(problem.fn, y, args)
        )
        return (self.minimiser.init(problem, y, args, options), y, f0)

    def step(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: PyTree,
        options: Dict[str, Any],
        state: PyTree,
    ):
        # Keep track of the running min and output this. keep track of the running
        # iterate through state instead and manage this by augmenting the auxiliary
        # output.
        minimiser_state, state_y, f_min = state
        problem_pipethrough = _ProblemPipeThrough(
            problem.fn, problem.has_aux, problem.tags
        )
        y_new, minimiser_state_new, (f_val, aux) = self.minimiser.step(
            problem_pipethrough, state_y, args, options, minimiser_state
        )
        new_best = f_val < f_min
        y_min = tree_where(new_best, y_new, y)
        f_min_new = jnp.where(new_best, f_val, f_min)
        new_state = (minimiser_state_new, y_new, f_min_new)
        return y_min, new_state, aux

    def terminate(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: PyTree,
        options: Dict[str, Any],
        state: PyTree,
    ):
        minimiser_state, state_y, _ = state
        return self.minimiser.terminate(
            problem, state_y, args, options, minimiser_state
        )

    def buffers(
        self,
        state: PyTree,
    ):
        minimiser_state, *_ = state
        return self.minimiser.buffers(minimiser_state)


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
    y0 = jtu.tree_map(jnp.asarray, y0)
    struct = jax.eval_shape(lambda: problem.fn(y0, args))
    if problem.has_aux:
        struct, aux_struct = struct
    else:
        aux_struct = None

    if not (isinstance(struct, jax.ShapeDtypeStruct) and struct.shape == ()):
        raise ValueError("minimisation function must output a single scalar.")

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
        aux_struct=aux_struct,
    )
