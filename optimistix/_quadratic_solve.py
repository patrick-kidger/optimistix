from typing import Any, cast, Generic, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import (
    Aux,
    Constraint,
    DescentState,
    EqualityOut,
    Fn,
    InequalityOut,
    MaybeAuxFn,
    Out,
    SearchState,
    Y,
)
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._misc import (
    cauchy_termination,
    checked_bounds,
    evaluate_constraint,
    filter_cond,
    inexact_asarray,
    lin_to_grad,
    NoneAux,
    OutAsArray,
    tree_full_like,
)
from ._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
    Iterate,
)
from ._solution import RESULTS, Solution


def _rewrite_quadratic(quadratic_minimum, _, inputs):
    quadratic_minimise_fn, _, _, fn_args, *_ = inputs
    del inputs

    def no_aux(x):
        f_val, _ = quadratic_minimise_fn(x, fn_args)
        return f_val

    return jax.grad(no_aux)(quadratic_minimum)


class _QuadraticSolverState(
    eqx.Module, Generic[Y, Aux, SearchState, DescentState], strict=True
):
    # Never updated
    hessian: lx.AbstractLinearOperator
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated every descent step
    f_info: FunctionInfo.EvalGradHessian
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS


class AbstractQuadraticSolver(
    AbstractIterativeSolver[Y, Iterate.Primal, Out, Aux, _QuadraticSolverState],
    strict=True,
):
    """Abstract base class for all quadratic solvers.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    descent: AbstractVar[
        AbstractDescent[Y, Iterate.Primal, FunctionInfo.EvalGradHessian, Any]
    ]
    search: AbstractVar[
        AbstractSearch[
            Y,
            FunctionInfo.EvalGradHessian,
            Union[
                FunctionInfo.Eval, FunctionInfo.EvalGrad, FunctionInfo.EvalGradHessian
            ],
            Any,
        ]
    ]

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> tuple[Iterate.Primal, _QuadraticSolverState]:
        # TODO: repeating this can probably be avoided (unless constraints nonlinear)
        if constraint is not None:
            evaluated = evaluate_constraint(constraint, y)
            constraint_residual, constraint_jacobians = evaluated
        else:
            constraint_residual = constraint_jacobians = None

        hessian, _ = jax.hessian(fn)(y, args)
        hessian = lx.PyTreeLinearOperator(hessian, jax.eval_shape(lambda: y))
        f_info = FunctionInfo.EvalGradHessian(
            jnp.zeros(f_struct.shape, f_struct.dtype),
            y,  # Shape equivalent with grad
            hessian,
            bounds,
            constraint_residual,
            constraint_jacobians,  # pyright: ignore  # TODO: fix this!
        )
        f_info_struct = jax.eval_shape(lambda: f_info)

        state = _QuadraticSolverState(
            hessian=hessian,
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),  # pyright: ignore TODO
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )
        return Iterate.Primal(y), state

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        iterate: Iterate.Primal,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _QuadraticSolverState,
        tags: frozenset[object],
    ) -> tuple[Iterate.Primal, _QuadraticSolverState, Aux]:  # TODO iterate type
        autodiff_mode = options.get("autodiff_mode", "bwd")
        y = iterate.y  # TODO: typing fix

        # TODO: there is no reason for this to be done again - the constraints are
        # assumed to be linear here. (Are they always?) -> No, not always
        if constraint is not None:
            evaluated = evaluate_constraint(constraint, y)
            constraint_residual, constraint_jacobians = evaluated
        else:
            constraint_residual = constraint_jacobians = None

        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )

        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval, bounds, constraint_residual),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)

            f_eval_info = FunctionInfo.EvalGradHessian(
                f_eval,
                grad,
                state.hessian,
                bounds,
                constraint_residual,
                constraint_jacobians,  # pyright: ignore  # TODO: fix this!
            )
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)

            # TODO(jhaffner): Are we taking one too many steps?
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step

            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        state = _QuadraticSolverState(
            hessian=state.hessian,  # Re-use
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
        )

        return Iterate.Primal(y), state, aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        iterate: Iterate.Primal,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _QuadraticSolverState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        iterate: Iterate.Primal,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _QuadraticSolverState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        # TODO(jhaffner): Implement any postprocessing?
        # Add set of constraints active at solution, values of dual variables?
        return iterate.y, aux, {}


# Keep `optx.implicit_jvp` happy.
if _rewrite_quadratic.__globals__["__name__"].startswith("jaxtyping"):
    _rewrite_quadratic = _rewrite_quadratic.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


@eqx.filter_jit
def quadratic_solve(
    fn: MaybeAuxFn[Y, Out, Aux],
    # no type parameters, see https://github.com/microsoft/pyright/discussions/5599
    solver: AbstractQuadraticSolver,
    y0: Y,
    args: PyTree[Any] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    constraint: Optional[Constraint[Y, EqualityOut, InequalityOut]] = None,
    bounds: Optional[tuple[Y, Y]] = None,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution[Y, Aux]:
    """Solve a quadratic problem.

    This minimises a `fn(y, args)` which returns a scalar value and is quadratic in `y`.

        **Arguments:**

    - `fn`: The residual function. This should take two arguments: `fn(y, args)` and
        return a scalar.
    - `solver`: The quadratic solver to use.
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
        any structure of the Hessian of `y -> fn(y, args)**2` with respect to y for
        unconstrained problems, or any structure of the KKT system for constrained
        problems.
        Used with [`optimistix.ImplicitAdjoint`][] to implement the implicit function
        theorem as efficiently as possible. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    if not has_aux:
        fn = NoneAux(fn)  # pyright: ignore
    fn = OutAsArray(fn)

    if not isinstance(solver, AbstractQuadraticSolver):
        raise NotImplementedError("No other solver types supported yet.")
    else:
        y0 = jtu.tree_map(inexact_asarray, y0)

        fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore
        fn = cast(Fn[Y, Out, Aux], fn)
        f_struct, aux_struct = fn.out_struct

        if bounds is not None:
            bounds = checked_bounds(y0, jtu.tree_map(inexact_asarray, bounds))

        if constraint is not None:
            constraint = OutAsArray(constraint)  # TODO repeat this in other APIs
            constraint = eqx.filter_closure_convert(constraint, y0)
            constraint = cast(Constraint[Y, EqualityOut, InequalityOut], constraint)

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
            rewrite_fn=_rewrite_quadratic,
        )
