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

from typing import Any, cast, Generic, Optional, Union

import equinox as eqx
import jax
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Args, Aux, Fn, MaybeAuxFn, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._misc import inexact_asarray, NoneAux
from ._root_find import AbstractRootFinder, root_find
from ._solution import Solution


class AbstractFixedPointSolver(AbstractIterativeSolver[SolverState, Y, Y, Aux]):
    pass


def _fixed_point(fixed_point, _, inputs):
    fixed_point_fn, args, *_ = inputs
    del inputs
    f_val, _ = fixed_point_fn(fixed_point, args)
    return (f_val**ω - fixed_point**ω).ω


class _ToRootFn(eqx.Module, Generic[Y, Aux]):
    fixed_point_fn: Fn[Y, Y, Aux]

    def __call__(self, y: Y, args: Args) -> tuple[Y, Aux]:
        out, aux = self.fixed_point_fn(y, args)
        if (
            eqx.tree_equal(jax.eval_shape(lambda: y), jax.eval_shape(lambda: out))
            is not True
        ):
            raise ValueError(
                "The input and output of `fixed_point_fn` must have the same structure"
            )
        return (out**ω - y**ω).ω, aux


@eqx.filter_jit
def fixed_point(
    fn: MaybeAuxFn[Y, Y, Aux],
    solver: Union[AbstractFixedPointSolver, AbstractRootFinder],
    y0: Y,
    args: PyTree[Any] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset()
) -> Solution[Y, Aux]:

    if not has_aux:
        fn = NoneAux(fn)
    fn = cast(Fn[Y, Y, Aux], fn)
    if isinstance(solver, AbstractRootFinder):
        del tags
        return root_find(
            _ToRootFn(fn),
            solver,
            y0,
            args,
            options,
            has_aux=True,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
    else:
        y0 = jtu.tree_map(inexact_asarray, y0)
        f_struct, aux_struct = jax.eval_shape(lambda: fn(y0, args))
        if eqx.tree_equal(jax.eval_shape(lambda: y0), f_struct) is not True:
            raise ValueError(
                "The input and output of `fixed_point_fn` must have the same structure"
            )
        return iterative_solve(
            fn,
            solver,
            y0,
            args,
            options,
            rewrite_fn=_fixed_point,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
            f_struct=f_struct,
            aux_struct=aux_struct,
        )
