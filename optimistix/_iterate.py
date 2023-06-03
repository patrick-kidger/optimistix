import abc
from typing import (
    Any,
    Callable,
    FrozenSet,
    Generic,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, PyTree

from ._adjoint import AbstractAdjoint
from ._custom_types import Aux, Fn, Out, SolverState, Y
from ._solution import RESULTS, Solution


if TYPE_CHECKING:
    _Node = Any
else:
    _Node = eqxi.doc_repr(Any, "Node")


class _AuxError:
    def __bool__(self):
        raise ValueError("")


aux_error = _AuxError()


def _is_jaxpr(x):
    return isinstance(x, (jax.core.Jaxpr, jax.core.ClosedJaxpr))


def _is_array_or_jaxpr(x):
    return _is_jaxpr(x) or eqx.is_array(x)


class AbstractIterativeSolver(eqx.Module, Generic[SolverState, Y, Out, Aux]):
    @abc.abstractmethod
    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> SolverState:
        ...

    @abc.abstractmethod
    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: SolverState,
        tags: frozenset[object],
    ) -> tuple[Y, SolverState, Aux]:
        ...

    @abc.abstractmethod
    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: SolverState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        ...

    @abc.abstractmethod
    def buffers(
        self,
        state: SolverState,
    ) -> Union[_Node, Sequence[_Node]]:
        ...


def _zero(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jnp.zeros(x.shape, dtype=x.dtype)
    else:
        return x


def _iterate(inputs, while_loop):
    fn, args, solver, y0, options, max_steps, f_struct, aux_struct, tags = inputs
    del inputs
    static_leaf = lambda x: isinstance(x, eqxi.Static)
    f_struct = jtu.tree_map(lambda x: x.value, f_struct, is_leaf=static_leaf)
    aux_struct = jtu.tree_map(lambda x: x.value, aux_struct, is_leaf=static_leaf)

    if options is None:
        options = {}

    init_aux = jtu.tree_map(_zero, aux_struct)
    init_state = solver.init(fn, y0, args, options, f_struct, aux_struct, tags)
    dynamic_init_state, static_state = eqx.partition(init_state, eqx.is_array)

    init_carry = (y0, 0, dynamic_init_state, init_aux)

    def cond_fun(carry):
        y, _, dynamic_state, _ = carry
        state = eqx.combine(static_state, dynamic_state)
        terminate, _ = solver.terminate(fn, y, args, options, state, tags)
        return jnp.invert(terminate)

    def body_fun(carry):
        y, num_steps, dynamic_state, _ = carry
        state = eqx.combine(static_state, dynamic_state)
        new_y, new_state, aux = solver.step(fn, y, args, options, state, tags)
        new_dynamic_state, new_static_state = eqx.partition(new_state, eqx.is_array)

        new_static_state_no_jaxpr = eqx.filter(
            new_static_state, _is_jaxpr, inverse=True
        )
        static_state_no_jaxpr = eqx.filter(state, _is_array_or_jaxpr, inverse=True)
        assert eqx.tree_equal(static_state_no_jaxpr, new_static_state_no_jaxpr) is True
        return new_y, num_steps + 1, new_dynamic_state, aux

    def buffers(carry):
        _, _, state, _ = carry
        return solver.buffers(state)

    final_carry = while_loop(
        cond_fun, body_fun, init_carry, max_steps=max_steps, buffers=buffers
    )

    final_y, num_steps, final_state, aux = final_carry
    _final_state = eqx.combine(static_state, final_state)
    terminate, result = solver.terminate(fn, final_y, args, options, _final_state, tags)
    result = RESULTS.where(
        (result == RESULTS.successful) & jnp.invert(terminate),
        RESULTS.max_steps_reached,
        result,
    )
    return final_y, (num_steps, result, final_state, aux)


def iterative_solve(
    fn: Fn[Y, Out, Aux],
    solver: AbstractIterativeSolver,
    y0: PyTree[Array],
    args: PyTree = None,
    options: Optional[dict[str, Any]] = None,
    *,
    rewrite_fn: Callable,
    max_steps: Optional[int],
    adjoint: AbstractAdjoint,
    throw: bool,
    tags: FrozenSet[object],
    f_struct: PyTree[jax.ShapeDtypeStruct],
    aux_struct: PyTree[jax.ShapeDtypeStruct],
) -> Solution:
    f_struct = jtu.tree_map(eqxi.Static, f_struct)
    aux_struct = jtu.tree_map(eqxi.Static, aux_struct)
    inputs = fn, args, solver, y0, options, max_steps, f_struct, aux_struct, tags
    out, (num_steps, result, final_state, aux) = adjoint.apply(
        _iterate, rewrite_fn, inputs, tags
    )
    stats = {"num_steps": num_steps, "max_steps": max_steps}
    sol = Solution(value=out, result=result, state=final_state, aux=aux, stats=stats)
    if throw:
        sol = result.error_if(sol, result != RESULTS.successful)
    return sol
