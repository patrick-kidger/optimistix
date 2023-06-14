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

from typing import Any, Callable, Optional, Union

import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, Out, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._minimise import AbstractMinimiser, minimise
from ._misc import inexact_asarray, NoneAux, sum_squares
from ._solution import Solution


class AbstractLeastSquaresSolver(AbstractIterativeSolver[SolverState, Y, Out, Aux]):
    pass


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
        return minimise(
            fn=_ToMinimiseFn(fn),
            solver=solver,
            y0=y0,
            tags=tags,
            args=args,
            options=options,
            has_aux=True,
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
