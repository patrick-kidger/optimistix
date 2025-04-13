from collections.abc import Callable
from typing import Any, Generic, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import (
    Aux,
    Constraint,
    DescentState,
    EqualityOut,
    Fn,
    InequalityOut,
    Out,
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
from .learning_rate import LearningRate


class _SteepestDescentState(eqx.Module, Generic[Y]):
    grad: Y


_FnInfo: TypeAlias = (
    FunctionInfo.EvalGrad
    | FunctionInfo.EvalGradHessian
    | FunctionInfo.EvalGradHessianInv
    | FunctionInfo.ResidualJac
)


class SteepestDescent(AbstractDescent[Y, _FnInfo, _SteepestDescentState]):
    """The descent direction given by locally following the gradient."""

    norm: Callable[[PyTree], Scalar] | None = None

    def init(self, y: Y, f_info_struct: _FnInfo) -> _SteepestDescentState:
        del f_info_struct
        # Dummy; unused
        return _SteepestDescentState(y)

    def query(
        self, y: Y, f_info: _FnInfo, state: _SteepestDescentState
    ) -> _SteepestDescentState:
        if isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
            ),
        ):
            grad = f_info.grad
        elif isinstance(f_info, FunctionInfo.ResidualJac):
            grad = f_info.compute_grad()
        else:
            raise ValueError(
                "Cannot use `SteepestDescent` with this solver. This is because "
                "`SteepestDescent` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )
        if self.norm is not None:
            grad = (grad**ω / self.norm(grad)).ω
        return _SteepestDescentState(grad)

    def step(
        self, step_size: Scalar, state: _SteepestDescentState
    ) -> tuple[Y, RESULTS]:
        return (-step_size * state.grad**ω).ω, RESULTS.successful


SteepestDescent.__init__.__doc__ = """**Arguments:**

- `norm`: If passed, then normalise the gradient using this norm. (The returned step
    will have length `step_size` with respect to this norm.) Optimistix includes three
    built-in norms: [`optimistix.max_norm`][], [`optimistix.rms_norm`][], and
    [`optimistix.two_norm`][].
"""


class _GradientDescentState(
    eqx.Module, Generic[Y, Out, Aux, SearchState, DescentState]
):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated after each descent step
    f_info: FunctionInfo.EvalGrad
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS


class AbstractGradientDescent(AbstractMinimiser[Y, Aux, _GradientDescentState]):
    """The gradient descent method for unconstrained minimisation.

    At every step, this algorithm performs a line search along the steepest descent
    direction. You should subclass this to provide it with a particular choice of line
    search. (E.g. [`optimistix.GradientDescent`][] uses a simple learning rate step.)

    Subclasses must provide the following abstract attributes, with the following types:

    - `rtol: float`
    - `atol: float`
    - `norm: Callable[[PyTree], Scalar]`
    - `descent: AbstractDescent`
    - `search: AbstractSearch`

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, FunctionInfo.EvalGrad, Any]]
    search: AbstractVar[
        AbstractSearch[Y, FunctionInfo.EvalGrad, FunctionInfo.Eval, Any]
    ]
    verbose: AbstractVar[frozenset[str]]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GradientDescentState:
        del constraint  # TODO jhaffner: these are not handled. Currently writing None
        # into FunctionInfo.Eval. This should be handled properly - raising errors, and
        # making things explicit in the documentation.
        # Currently, Eval requests constraint residual information, but EvalGrad does
        # not. I have sprinkled TODO notes liberally, so I should be able to find all
        # the places that need to be updated.
        f_info = FunctionInfo.EvalGrad(
            jnp.zeros(f_struct.shape, f_struct.dtype), y, bounds
        )
        f_info_struct = jax.eval_shape(lambda: f_info)
        return _GradientDescentState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _GradientDescentState,
        tags: frozenset[object],
    ) -> tuple[Y, _GradientDescentState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            # TODO: constraint_residual! FunctionInfo.Eval now requires constraint
            # information. This solver should specify to what extent, if any, it handles
            # constraints. If gradient methods do not deal with constraints, then this
            # should be added here as a comment.
            FunctionInfo.Eval(f_eval, bounds, None),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)

            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad, bounds)
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)
            y_diff = (state.y_eval**ω - y**ω).ω
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
        y_eval = (y**ω + y_descent**ω).ω
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        state = _GradientDescentState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
        )
        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _GradientDescentState,
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
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _GradientDescentState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class GradientDescent(AbstractGradientDescent[Y, Aux]):
    """Classic gradient descent with a learning rate `learning_rate`.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: SteepestDescent[Y]
    search: LearningRate[Y]
    verbose: frozenset[str]

    def __init__(
        self,
        learning_rate: float,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = SteepestDescent()
        self.search = LearningRate(learning_rate)
        self.verbose = verbose


GradientDescent.__init__.__doc__ = """**Arguments:**

- `learning_rate`: Specifies a constant learning rate to use at each step.
- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `loss`, `step_size` and `y`. For example 
    `verbose=frozenset({"loss"})`.
"""
