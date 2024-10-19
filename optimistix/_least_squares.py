from typing import Any, cast, Generic, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree, Scalar

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import (
    Args,
    Aux,
    Constraint,
    ConstraintOut,
    Fn,
    MaybeAuxFn,
    Out,
    SolverState,
    Y,
)
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._minimise import AbstractMinimiser, minimise
from ._misc import (
    checked_bounds,
    inexact_asarray,
    NoneAux,
    OutAsArray,
    sum_squares,
)
from ._solution import Solution


class AbstractLeastSquaresSolver(
    AbstractIterativeSolver[Y, Out, Aux, SolverState], strict=True
):
    """Abstract base class for all least squares solvers."""


def _rewrite_fn(optimum, _, inputs):
    residual_fn, _, _, args, *_ = inputs
    del inputs

    def objective(_optimum):
        residual, _ = residual_fn(_optimum, args)
        return 0.5 * sum_squares(residual)

    return jax.grad(objective)(optimum)


# Keep `optx.implicit_jvp` is happy.
if _rewrite_fn.__globals__["__name__"].startswith("jaxtyping"):
    _rewrite_fn = _rewrite_fn.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


class _ToMinimiseFn(eqx.Module, Generic[Y, Out, Aux]):
    residual_fn: Fn[Y, Out, Aux]

    def __call__(self, y: Y, args: Args) -> tuple[Scalar, Aux]:
        residual, aux = self.residual_fn(y, args)
        return 0.5 * sum_squares(residual), aux


# TODO: Leave keyword argument for constraints if least squares solvers don't handle
# these? See fixed_point for a more thorough description of the choice to make here.
@eqx.filter_jit
def least_squares(
    fn: MaybeAuxFn[Y, Out, Aux],
    # no type parameters, see https://github.com/microsoft/pyright/discussions/5599
    solver: Union[AbstractLeastSquaresSolver, AbstractMinimiser],
    y0: Y,
    args: PyTree[Any] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    constraint: Optional[Constraint[Y, ConstraintOut]] = None,
    bounds: Optional[tuple[Y, Y]] = None,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution[Y, Aux]:
    r"""Solve a nonlinear least-squares problem.

    Given a nonlinear function `fn(y, args)` which returns a pytree of residuals,
    this returns the solution to $\min_y \sum_i \textrm{fn}(y, \textrm{args})_i^2$.

    **Arguments:**

    - `fn`: The residual function. This should take two arguments: `fn(y, args)` and
        return a pytree of arrays not necessarily of the same shape as the input `y`.
    - `solver`: The least-squares solver to use. This can be either an
        [`optimistix.AbstractLeastSquaresSolver`][] solver, or an
        [`optimistix.AbstractMinimiser`][]. If `solver` is an
        [`optimistix.AbstractMinimiser`][], then it will attempt to minimise the scalar
        loss $\min_y \sum_i \textrm{fn}(y, \textrm{args})_i^2$ directly.
    - `y0`: An initial guess for what `y` may be.
    - `args`: Passed as the `args` of `fn(y, args)`.
    - `options`: Individual solvers may accept additional runtime arguments.
        See each individual solver's documentation for more details.
    - `has_aux`: If `True`, then `fn` may return a pair, where the first element is its
        function value, and the second is just auxiliary data. Keyword only argument.
    - `constraint`: Individual solvers may accept a constraint function `constraint(y)`.
        This must return a PyTree of scalars, one for each constraint, such that
        `constraint(y) >= 0` if the constraint is satisfied. The initial value `y0` must
        be feasible with respect to these constraints. Keyword only argument.
    - `bounds`: Individual solvers may accept bounds. This should be a pair of pytrees
        of the same structure as `y`, where the first element is the lower bound, and
        the second is the upper bound. Unbounded leaves can be indicated with +/- inf
        for the upper and lower bounds, respectively. For finite bounds, it is checked
        whether `y` is in the closed interval `[lower, upper]`. Keyword only argument.
    - `max_steps`: The maximum number of steps the solver can take. Keyword only
        argument.
    - `adjoint`: The adjoint method used to compute gradients through the fixed-point
        solve. Keyword only argument.
    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or encountering divergent iterates.) If `True` then a failure will raise
        an error. If `False` then the returned solution object will have a `result`
        field indicating whether any failures occured. (See [`optimistix.Solution`][].)
        Keyword only argument.
    - `tags`: Lineax [tags](https://docs.kidger.site/lineax/api/tags/) describing
        any structure of the Hessian of `y -> sum(fn(y, args)**2)` with respect to y.
        Used with [`optimistix.ImplicitAdjoint`][] to implement the implicit function
        theorem as efficiently as possible. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    if not has_aux:
        fn = NoneAux(fn)  # pyright: ignore
    fn = OutAsArray(fn)

    if isinstance(solver, AbstractMinimiser):
        del tags
        return minimise(
            _ToMinimiseFn(fn),
            solver,
            y0,
            args,
            options,
            has_aux=True,
            constraint=constraint,
            bounds=bounds,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
    else:
        y0 = jtu.tree_map(inexact_asarray, y0)
        fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore
        fn = cast(Fn[Y, Out, Aux], fn)
        f_struct, aux_struct = fn.out_struct

        if bounds is not None:
            bounds = checked_bounds(y0, jtu.tree_map(inexact_asarray, bounds))

        if constraint is not None:
            constraint = eqx.filter_closure_convert(constraint, y0)
            constraint = cast(Constraint[Y, ConstraintOut], constraint)
            # TODO(jhaffner): this can be done more elegantly
            msg = "The initial point must be feasible."
            pred = jnp.all(constraint(y0) >= 0)  # pyright: ignore (ConstraintOut array)
            eqx.error_if(y0, pred, msg)

        return iterative_solve(
            fn,
            solver,
            y0,
            args,
            options,
            constraint=constraint,
            bounds=bounds,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
            f_struct=f_struct,
            aux_struct=aux_struct,
            rewrite_fn=_rewrite_fn,
        )
