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

from typing import Any, Optional

import equinox as eqx
import jax
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, Out, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._misc import NoneAux
from ._solution import Solution


class AbstractRootFinder(AbstractIterativeSolver[SolverState, Y, Out, Aux]):
    pass


def _root(root, _, inputs):
    root_fn, args, *_ = inputs
    del inputs
    out, _ = root_fn(root, args)
    return out


@eqx.filter_jit
def root_find(
    fn: Fn[Y, Out, Aux],
    solver: AbstractRootFinder,
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

    if not has_aux:
        fn = NoneAux(fn)

    f_struct, aux_struct = jax.eval_shape(lambda: fn(y0, args))

    return iterative_solve(
        fn,
        solver,
        y0,
        args,
        options,
        rewrite_fn=_root,
        max_steps=max_steps,
        adjoint=adjoint,
        throw=throw,
        tags=tags,
        aux_struct=aux_struct,
        f_struct=f_struct,
    )