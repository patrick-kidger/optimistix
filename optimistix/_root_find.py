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

from typing import Any, cast, Optional

import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, MaybeAuxFn, Out, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._misc import inexact_asarray, NoneAux
from ._solution import Solution


class AbstractRootFinder(AbstractIterativeSolver[SolverState, Y, Out, Aux]):
    """Abstract base class for all root finders."""


def _root(root, _, inputs):
    root_fn, args, *_ = inputs
    del inputs
    f_val, _ = root_fn(root, args)
    return f_val


@eqx.filter_jit
def root_find(
    fn: MaybeAuxFn[Y, Out, Aux],
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
) -> Solution[Y, Aux]:
    """Solve a root-finding problem.

    Given a nonlinear function `fn(y, args)` which returns a pytree of arrays,
    this returns the value `z` such that `fn(z, args) = 0`.

    **Arguments:**

    - `fn`: The function to find the roots of. This should take two arguments:
        `fn(y, args)` and return a pytree of arrays not necessarily of the same shape
        as the input `y`.
    - `solver`: The root-finder to use. This should be an
        [`optimistix.AbstractRootFinder`][].
    - `y0`: An initial guess for what `y` may be.
    - `args`: Passed as the `args` of `fn(y, args)`.
    - `options`: Individual solvers may accept additional runtime arguments.
        See each individual solver's documentation for more details.
    - `has_aux`: If `True`, then `fn` may return a pair, where the first element is its
        function value, and the second is just auxiliary data. Keyword only argument.
    - `max_steps`: The maximum number of steps the solver can take. Keyword only
        argument.
    - `adjoint`: The adjoint method used to compute gradients through the fixed-point
        solve. Keyword only argument.
    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or encountering divergent iterates.) If `True` then a failure will raise
        an error. If `False` then the returned solution object will have a `result`
        field indicating whether any failures occured. (See [`optimistix.Solution`][].)
        Keyword only argument.
    - `tags`: Lineax [tags](https://docs.kidger.site/lineax/api/tags/) describing the
        any structure of the Jacobian of `fn` with respect to `y`. Used with
        [`optimistix.ImplicitAdjoint`][] to implement the implicit function theorem as
        efficiently as possible. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    y0 = jtu.tree_map(inexact_asarray, y0)
    if not has_aux:
        fn = NoneAux(fn)
    fn = cast(Fn[Y, Out, Aux], fn)
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
        f_struct=f_struct,
        aux_struct=aux_struct,
    )
