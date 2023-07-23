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
from jaxtyping import PyTree, Scalar

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._base_solver import AbstractSolver
from ._custom_types import Args, Aux, Fn, MaybeAuxFn, Out, SolverState, Y
from ._minimise import AbstractMinimiser, minimise
from ._misc import inexact_asarray, NoneAux, sum_squares
from ._solution import Solution


class AbstractLeastSquaresSolver(AbstractSolver[Y, Out, Aux, SolverState]):
    """Abstract base class for all least squares solvers."""

    @staticmethod
    def rewrite_fn(optimum, _, inputs):
        _, residual_fn, args, *_ = inputs
        del inputs

        def objective(_optimum):
            residual, _ = residual_fn(_optimum, args)
            return sum_squares(residual)

        return jax.grad(objective)(optimum)


class _ToMinimiseFn(eqx.Module, Generic[Y, Out, Aux]):
    residual_fn: Fn[Y, Out, Aux]

    def __call__(self, y: Y, args: Args) -> tuple[Scalar, Aux]:
        residual, aux = self.residual_fn(y, args)
        return 0.5 * sum_squares(residual), aux


@eqx.filter_jit
def least_squares(
    fn: MaybeAuxFn[Y, Out, Aux],
    solver: Union[
        AbstractLeastSquaresSolver[Y, Out, Aux, SolverState],
        AbstractMinimiser[Y, Aux, SolverState],
    ],
    y0: Y,
    args: PyTree[Any] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution[Y, Aux, SolverState]:
    r"""Solve a nonlinear least-squares problem.

    Given a nonlinear function `fn(y, args)` which returns a pytree of residuals,
    this returns the solution to $\min_y \sum_i \textrm{fn}(y, \textrm{args})_i^2$.

    **Arguments:**

    - `fn`: The residual function. This should take two arguments: `fn(y, args)` and
        return a pytree of arrays not necessarily of the same shape as the input `y`.
    - `solver`: The least-squares solver to use. This can be either an
        [`optimistix.AbstractLeastSquares`][] solver, or an
        [`optimistix.AbstractMinimiser`][]. If `solver` is an
        [`AbstractMinimiser`][], then it will attempt to minimise the scalar loss
        $\min_y \sum_i \textrm{fn}(y, \textrm{args})_i^2$ directly.
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
        any structure of the Hessian of `y -> sum(fn(y, args)**2)` with respect to y.
        Used with [`optimistix.ImplicitAdjoint`][] to implement the implicit function
        theorem as efficiently as possible. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    if not has_aux:
        fn = NoneAux(fn)
    fn = cast(Fn[Y, Out, Aux], fn)
    if options is None:
        options = {}

    if isinstance(solver, AbstractMinimiser):
        del tags
        return minimise(
            _ToMinimiseFn(fn),
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
        return solver.solve(
            fn,
            y0,
            args,
            options,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
            f_struct=f_struct,
            aux_struct=aux_struct,
        )
