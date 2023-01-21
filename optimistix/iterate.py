import abc
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .adjoint import AbstractAdjoint
from .linear_operator import Pattern
from .misc import NoneAux
from .solution import RESULTS, Solution


_SolverState = TypeVar("_SolverState")
_Aux = TypeVar("_Aux")


class AbstractIterativeProblem(eqx.Module):
    fn: Callable
    has_aux: bool


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
    ) -> Tuple[PyTree[Array], _SolverState, _Aux]:
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


def _zero(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jnp.zeros(x.shape, dtype=x.dtype)
    else:
        return x


def _iterate(inputs, closure, while_loop):
    prob, args = inputs
    solver, y0, options, max_steps = closure
    del inputs, closure

    if options is None:
        options = {}

    init_state = solver.init(prob, y0, args, options)

    if prob.has_aux:

        def _aux(_solver, _prob, _y, _args, _options, _state):
            _, _, _aux = _solver.step(_prob, _y, _args, _options, _state)
            return _aux

        init_aux = eqx.filter_eval_shape(
            _aux, solver, prob, y0, args, options, init_state
        )
        init_aux = jtu.tree_map(_zero, init_aux)
    else:
        prob = eqx.tree_at(lambda p: p.fn, prob, NoneAux(prob.fn))
        init_aux = None

    init_val = (y0, 0, init_state, init_aux)

    def cond_fun(carry):
        y, _, state, _ = carry
        terminate, _ = solver.terminate(prob, y, args, options, state)
        return jnp.invert(terminate)

    def body_fun(carry, _):
        y, num_steps, state, _ = carry
        new_y, new_state, aux = solver.step(prob, y, args, options, state)
        return new_y, num_steps + 1, new_state, aux

    final_val = while_loop(cond_fun, body_fun, init_val, max_steps)

    final_y, num_steps, final_state, aux = final_val
    terminate, result = solver.terminate(prob, final_y, args, final_state, options)
    result = jnp.where(
        (result == RESULTS.successful) & jnp.invert(terminate),
        RESULTS.max_steps_reached,
        result,
    )
    return final_y, (num_steps, result, final_state, aux)


def iterative_solve(
    prob: AbstractIterativeProblem,
    solver: AbstractIterativeSolver,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[Dict[str, Any]] = None,
    *,
    rewrite_fn: Callable,
    max_steps: Optional[int],
    adjoint: AbstractAdjoint,
    throw: bool,
    pattern: Pattern,
) -> Solution:
    inputs = prob, args
    closure = solver, y0, options, max_steps
    out, (num_steps, result, final_state, aux) = adjoint.apply(
        _iterate, rewrite_fn, inputs, closure, pattern, max_steps
    )
    stats = {"num_steps": num_steps, "max_steps": max_steps}
    sol = Solution(value=out, result=result, state=final_state, aux=aux, stats=stats)
    sol = eqxi.branched_error_if(
        sol,
        throw & (result != RESULTS.successful),
        result,
        RESULTS.reverse_lookup,
    )
    return sol
