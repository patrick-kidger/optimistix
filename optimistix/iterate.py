import abc
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint, ImplicitAdjoint
from .linear_operator import Pattern
from .results import RESULTS


_SolverState = TypeVar("_SolverState")


class AbstractIterativeProblem(eqx.Module):
    pass


class AbstractIterativeSolver(eqx.Module):
    @abc.abstractmethod
    def init(
        self,
        prob: AbstractIterativeProblem,
        y: PyTree[Array],
        args: PyTree,
        options: Dict[str, Any],
    ) -> _SolverState:
        ...

    @abc.abstractmethod
    def step(
        self,
        prob: AbstractIterativeProblem,
        y: PyTree[Array],
        args: PyTree,
        options: Dict[str, Any],
        state: _SolverState,
    ) -> Tuple[PyTree[Array], _SolverState]:
        ...

    @abc.abstractmethod
    def terminate(
        self,
        prob: AbstractIterativeProblem,
        y: PyTree[Array],
        args: PyTree,
        options: Dict[str, Any],
        state: _SolverState,
    ) -> bool:
        ...


def _iterate(inputs, closure, reverse_autodiffable):
    prob, args = inputs
    solver, y0, options, max_steps = closure
    del inputs, closure

    if options is None:
        options = {}

    init_state = solver.init(prob, y0, args, options)
    init_val = (y0, 0, init_state)

    def cond_fun(carry):
        y, _, state = carry
        terminate, _ = solver.terminate(prob, y, args, options, state)
        return jnp.invert(terminate)

    def body_fun(carry, _):
        y, num_steps, state = carry
        new_y, new_state = solver.step(prob, y, args, options, state)
        return new_y, num_steps + 1, new_state

    if reverse_autodiffable:
        final_val = eqxi.bounded_while_loop(
            cond_fun, body_fun, init_val, max_steps, base=4
        )
    else:
        if max_steps is None:
            _cond_fun = cond_fun
        else:

            def _cond_fun(carry):
                _, num_steps, _ = carry
                return cond_fun(carry) & (num_steps < max_steps)

        final_val = eqxi.bounded_while_loop(
            cond_fun, body_fun, init_val, max_steps=None
        )

    final_y, num_steps, final_state = final_val
    terminate, result = solver.terminate(prob, final_y, args, final_state, options)
    result = jnp.where(
        result == RESULTS.successful,
        jnp.where(terminate, RESULTS.successful, RESULTS.max_steps_reached),
        result,
    )
    return final_y, (num_steps, result, final_state)


def iterate_solve(
    prob: AbstractIterativeProblem,
    solver: AbstractIterativeSolver,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    rewrite_fn: Callable,
    max_steps: Optional[int] = 16,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    pattern: Pattern = Pattern(),
):
    inputs = prob, args
    closure = solver, y0, options, max_steps
    out, (num_steps, result, final_state) = adjoint.apply(
        _iterate, rewrite_fn, inputs, closure, pattern
    )
    stats = {"num_steps": num_steps, "max_steps": max_steps}
    outputs = out, result, final_state, stats

    error_index = eqxi.unvmap_max(result)
    outputs = eqxi.branched_error_if(
        outputs,
        throw & (result != RESULTS.successful),
        error_index,
        RESULTS.reverse_lookup,
    )
    return outputs
