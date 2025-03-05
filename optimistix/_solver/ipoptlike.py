from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import (
    Aux,
    Constraint,
    ConstraintOut,
    DescentState,
    Fn,
    SearchState,
    Y,
)
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    filter_cond,
    lin_to_grad,
    max_norm,
    tree_full_like,
    verbose_print,
)
from .._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .bfgs import BFGS, bfgs_update, identity_pytree
from .boundary_maps import ClosestFeasiblePoint
from .filtered import FilteredLineSearch
from .interior_point import InteriorDescent


_Hessian = TypeVar(
    "_Hessian", FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv
)


class _IPOPTLikeState(
    eqx.Module, Generic[Y, Aux, SearchState, DescentState, _Hessian], strict=True
):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated after each descent step
    f_info: _Hessian
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS
    # Used in compat.py
    num_accepted_steps: Int[Array, ""]


# TODO documentation
class AbstractIPOPTLike(
    AbstractMinimiser[Y, Aux, _IPOPTLikeState], Generic[Y, Aux, _Hessian], strict=True
):
    """Abstract IPOPT-like solver. Uses a filtered line search and an interior descent,
    and restores feasibility by solving a nonlinear subproblem if required. Approximates
    the Hessian using BFGS updates, as in [`optimistix.BFGS`][].

    This abstract version may be subclassed to choose alternative descent and searches.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, _Hessian, Any]]
    search: AbstractVar[AbstractSearch[Y, _Hessian, FunctionInfo.Eval, Any]]
    verbose: AbstractVar[frozenset[str]]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, ConstraintOut], None],
        bounds: Union[tuple[Y, Y], None],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _IPOPTLikeState:
        if constraint is not None:
            constraint_residual = constraint(y)
            constraint_bound = constraint(tree_full_like(y, 0))
            jac = jax.jacfwd(constraint)(y)  # TODO: options fwd, bwd
            out_structure = jax.eval_shape(lambda: constraint_residual)
            constraint_jac = lx.PyTreeLinearOperator(jac, out_structure)
        else:
            constraint_residual = None
            constraint_bound = None
            constraint_jac = None

        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        hessian = identity_pytree(y)
        f_info = FunctionInfo.EvalGradHessian(
            f,
            grad,
            hessian,
            y,
            bounds,
            constraint_residual,
            constraint_bound,
            constraint_jac,
        )
        f_info_struct = eqx.filter_eval_shape(lambda: f_info)
        return _IPOPTLikeState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, ConstraintOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _IPOPTLikeState,
        tags: frozenset[object],
    ) -> tuple[Y, _IPOPTLikeState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")

        if bounds is not None:
            lower, upper = bounds

        if constraint is not None:
            constraint_residual = constraint(state.y_eval)
            constraint_bound = constraint(tree_full_like(y, 0))
            jac = jax.jacfwd(constraint)(state.y_eval)  # TODO: options fwd, bwd
            out_structure = jax.eval_shape(lambda: constraint_residual)
            constraint_jac = lx.PyTreeLinearOperator(jac, out_structure)
        else:
            constraint_residual = None
            constraint_bound = None
            constraint_jac = None

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

            y_diff = (state.y_eval**ω - y**ω).ω

            hessian = state.f_info.hessian
            hessian_inv = None

            f_eval_info = bfgs_update(
                f_eval,
                grad,
                state.f_info.grad,
                hessian,
                hessian_inv,
                state.y_eval,
                y_diff,
                bounds,
                constraint_residual,
                constraint_bound,
                constraint_jac,
            )

            descent_state = self.descent.query(
                state.y_eval,
                f_eval_info,  # pyright: ignore
                descent_state,
            )
            f_diff = f_eval - state.f_info.f
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

        if len(self.verbose) > 0:
            verbose_loss = "loss" in self.verbose
            verbose_step_size = "step_size" in self.verbose
            verbose_y = "y" in self.verbose
            loss_eval = f_eval
            loss = state.f_info.f
            verbose_print(
                (verbose_loss, "Loss on this step", loss_eval),
                (verbose_loss, "Loss on the last accepted step", loss),
                (verbose_step_size, "Step size", step_size),
                (verbose_y, "y", state.y_eval),
                (verbose_y, "y on the last accepted step", y),
            )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        requires_restoration = (
            search_result == RESULTS.feasibility_restoration_required
        ) | (descent_result == RESULTS.feasibility_restoration_required)

        def restore(args):
            del args

            # TODO: make attribute and figure out how to update penalty parameter
            boundary_map = ClosestFeasiblePoint(
                1e-4, BFGS(rtol=1e-3, atol=1e-6, use_inverse=False)
            )
            recovered_y, restoration_result = boundary_map(
                state.y_eval, constraint, bounds
            )
            # TODO: what happens if the restoration requires a rescue itself?
            # TODO: Allow feasibility restoration to raise a certificate of
            # infeasibility and error out.
            return recovered_y, restoration_result

        def regular_update(args):
            search_result, descent_result = args
            result = RESULTS.where(
                search_result == RESULTS.successful, descent_result, search_result
            )
            y_eval = (y**ω + y_descent**ω).ω
            return y_eval, result

        args = (search_result, descent_result)
        y_eval, result = filter_cond(
            requires_restoration, restore, regular_update, args
        )

        state = _IPOPTLikeState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
        )
        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, ConstraintOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _IPOPTLikeState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, ConstraintOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _IPOPTLikeState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


# TODO: Edit docstring - this needs to be expanded quite a bit
class IPOPTLike(AbstractIPOPTLike[Y, Aux, _Hessian], strict=True):
    """An IPOPT-like solver. Uses a filtered line search and an interior descent, and
    restores infeasible steps by solving a nonlinear subproblem if required.

    Approximates the Hessian using BFGS updates, as in [`optimistix.BFGS`][].

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: InteriorDescent
    search: FilteredLineSearch
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = InteriorDescent()
        self.search = FilteredLineSearch()
        self.verbose = verbose


IPOPTLike.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example 
    `verbose=frozenset({"step_size", "loss"})`.
"""
