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

import abc
from typing import Any, cast, Generic, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree, Scalar

from ._adjoint import RecursiveCheckpointAdjoint
from ._base_solver import AbstractSolver
from ._custom_types import Aux, Fn, LineSearchState, MaybeAuxFn, Y
from ._misc import inexact_asarray, NoneAux
from ._solution import RESULTS, Solution


class AbstractDescent(eqx.Module, Generic[Y]):
    """The abstract base class for descents. A descent is a method which consumes a
    scalar, and returns the `diff` to take at point `y`, so that `y + diff` is the next
    iterate in a nonlinear optimisation problem.

    This generalises the concept of line search and trust region to anything which
    takes a step-size and returns the step to take given that step-size.
    """

    @abc.abstractmethod
    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        """Computes the descent direction.

        !!! warning

            This API is not stable and may change in the future.

            Right now the `options` dictionary is essentially unstructured, but figuring
            out a nicer type hierarachy to describe things seems pretty complicated. We
            may or may not do this in the future.

        **Arguments:**

        - `step_size`: a non-negative scalar describing the step size to take.
        - `args`: as passed to e.g. `optx.least_squares(..., args=...)`. Is just
            forwarded on to the user function.
        - `options`: an unstructured dictionary of "everything else" that the caller
            makes available for the descent, and which the descent can use to determine
            its output.

        **Returns:**

        The step to take in y-space.
        """


class AbstractLineSearch(AbstractSolver[Y, Scalar, Aux, LineSearchState]):
    """The abstract base class for all line searches."""

    @staticmethod
    def rewrite_fn(_, __, ___):
        assert False


@eqx.filter_jit
def line_search(
    fn: MaybeAuxFn[Y, Scalar, Aux],
    solver: AbstractLineSearch[Y, Aux, LineSearchState],
    y0: Y,
    args: PyTree[Any] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    max_steps: Optional[int] = 256,
    throw: bool = True,
    tags: frozenset = frozenset(),
) -> Solution[Y, Aux, LineSearchState]:
    """Perform a line search.

    This is essentially equivalent to [`optimistix.minimise`][], excpet there is no
    guarantee that the value will decrease, or that a local optimum is attained.

    **Arguments:**

    - `fn`: The objective function. This should take two arguments: `fn(y, args)` and
        return a scalar.
    - `solver`: The minimiser solver to use. This should be an
        [`optimistix.AbstractLineSearch`][].
    - `y0`: An initial guess for what `y` may be.
    - `args`: Passed as the `args` of `fn(y, args)`.
    - `options`: Individual solvers may accept additional runtime arguments.
        See each individual solver's documentation for more details.
    - `has_aux`: If `True`, then `fn` may return a pair, where the first element is its
        function value, and the second is just auxiliary data. Keyword only argument.
    - `max_steps`: The maximum number of steps the solver can take. Keyword only
        argument.
    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or encountering divergent iterates.) If `True` then a failure will raise
        an error. If `False` then the returned solution object will have a `result`
        field indicating whether any failures occured. (See [`optimistix.Solution`][].)
        Keyword only argument.
    - `tags`: Lineax [tags](https://docs.kidger.site/lineax/api/tags/) describing the
        any structure of the Hessian of `fn` with respect to `y`. Used in some solvers
        to improve efficiency. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    y0 = jtu.tree_map(inexact_asarray, y0)
    if not has_aux:
        fn = NoneAux(fn)
    fn = cast(Fn[Y, Scalar, Aux], fn)
    f_struct, aux_struct = jax.eval_shape(lambda: fn(y0, args))
    if options is None:
        options = {}

    if not (
        isinstance(f_struct, jax.ShapeDtypeStruct)
        and f_struct.shape == ()
        and jnp.issubdtype(f_struct.dtype, jnp.floating)
    ):
        raise ValueError(
            "line search function must output a single floating-point scalar."
        )

    return solver.solve(
        fn,
        y0,
        args,
        options,
        max_steps=max_steps,
        # Not implicit adjoint as we usually don't hit the minima.
        adjoint=RecursiveCheckpointAdjoint(),
        throw=throw,
        tags=tags,
        aux_struct=aux_struct,
        f_struct=f_struct,
    )
