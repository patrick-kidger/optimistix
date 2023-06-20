# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from typing import Any, cast, Optional, Union

import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, MaybeAuxFn, Out, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._minimise import AbstractMinimiser
from ._misc import inexact_asarray, NoneAux, sum_squares
from ._solution import Solution


class AbstractLeastSquaresSolver(AbstractIterativeSolver[SolverState, Y, Out, Aux]):
    pass


def _minimum(minimum, _, inputs):
    minimise_fn, args, *_ = inputs
    del inputs

    def min_no_aux(x):
        out, _ = minimise_fn(x, args)
        return out

    return jax.grad(min_no_aux)(minimum)


def _residual(optimum, _, inputs):
    residual_fn, args, *_ = inputs
    del inputs

    def objective(_optimum):
        residual, _ = residual_fn(_optimum, args)
        return sum_squares(residual)

    return jax.grad(objective)(optimum)


class _ToMinimiseFn(eqx.Module):
    residual_fn: Callable

    def __call__(self, y, args):
        residual, aux = self.residual_fn(y, args)
        return sum_squares(residual), aux


@eqx.filter_jit
def least_squares(
    fn: MaybeAuxFn[Y, Out, Aux],
    solver: Union[AbstractLeastSquaresSolver, AbstractMinimiser],
    y0: Y,
    args: PyTree[Any] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution[Y, Aux]:

    y0 = jtu.tree_map(inexact_asarray, y0)
    if not has_aux:
        fn = NoneAux(fn)
    fn = cast(Fn[Y, Out, Aux], fn)
    f_struct, aux_struct = jax.eval_shape(lambda: fn(y0, args))

    if isinstance(solver, AbstractMinimiser):
        return iterative_solve(
            fn=_ToMinimiseFn(fn),
            solver=solver,
            y0=y0,
            args=args,
            options=options,
            rewrite_fn=_minimum,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
            f_struct=f_struct,
            aux_struct=aux_struct,
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
            f_struct=f_struct,
            aux_struct=aux_struct,
        )
