from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree, Scalar

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._misc import NoneAux
from ._solution import Solution


class AbstractMinimiser(AbstractIterativeSolver[SolverState, Y, Scalar, Aux]):
    pass


def _minimum(optimum, _, inputs):
    minimise_fn, args, *_ = inputs
    del inputs

    def min_no_aux(x):
        out, _ = minimise_fn(x, args)
        return out

    return jax.grad(min_no_aux)(optimum)


@eqx.filter_jit
def minimise(
    fn: Fn[Y, Scalar, Aux],
    solver: AbstractMinimiser,
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
    y0 = jtu.tree_map(jnp.asarray, y0)

    if not has_aux:
        fn = NoneAux(fn)

    f_struct, aux_struct = jax.eval_shape(lambda: fn(y0, args))

    if not (isinstance(f_struct, jax.ShapeDtypeStruct) and f_struct.shape == ()):
        raise ValueError("minimisation function must output a single scalar.")

    return iterative_solve(
        fn,
        solver,
        y0,
        args,
        options,
        rewrite_fn=_minimum,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=throw,
        tags=tags,
        aux_struct=aux_struct,
        f_struct=f_struct,
    )
