from typing import Any, Callable, Optional, Union

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, Out, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._minimise import AbstractMinimiser, minimise
from ._misc import inexact_asarray, NoneAux
from ._solution import Solution


class AbstractLeastSquaresSolver(AbstractIterativeSolver[SolverState, Y, Out, Aux]):
    pass


def _residual(optimum, _, inputs):
    residual_fn, args, *_ = inputs
    del inputs

    def objective(_optimum):
        residual, _ = residual_fn(_optimum, args)
        sum_squared = jtu.tree_map(lambda x: jnp.sum(x**2), residual)
        return jtu.tree_reduce(lambda x, y: x + y, sum_squared)

    return jax.grad(objective)(optimum)


class _ToMinimiseFn(eqx.Module):
    residual_fn: Callable

    def __call__(self, y, args):
        out = self.residual_fn(y, args)
        out, aux = out
        out_ravel, _ = jfu.ravel_pytree(out)
        return jnp.sum(out_ravel**2), aux


@eqx.filter_jit
def least_squares(
    fn: Fn[Y, Out, Aux],
    solver: Union[AbstractLeastSquaresSolver, AbstractMinimiser],
    y0: Y,
    args: PyTree = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution:
    y0 = jtu.tree_map(inexact_asarray, y0)

    if not has_aux:
        fn = NoneAux(fn)

    f_struct, aux_struct = jax.eval_shape(lambda: fn(y0, args))

    if isinstance(solver, AbstractMinimiser):
        minimise_fn = _ToMinimiseFn(fn)
        return minimise(
            fn=minimise_fn,
            has_aux=True,
            solver=solver,
            y0=y0,
            tags=tags,
            args=args,
            options=options,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )

    else:
        return iterative_solve(
            fn,
            solver,
            y0,
            args,
            options,
            rewrite_fn=_residual,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
            aux_struct=aux_struct,
            f_struct=f_struct,
        )
