from collections.abc import Callable
from typing import Any, cast, Generic

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from .._custom_types import Aux, Y
from .._misc import (
    max_norm,
    sum_squares,
    tree_dot,
    tree_full_like,
    tree_where,
)
from .._search import AbstractDescent, AbstractSearch, FunctionInfo
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gradient_methods import AbstractGradientDescent


def polak_ribiere(grad_vector: Y, grad_prev: Y, y_diff_prev: Y) -> Scalar:
    """The Polak--Ribière formula for β. Used with [`optimistix.NonlinearCG`][] and
    [`optimistix.NonlinearCGDescent`][]."""
    del y_diff_prev
    numerator = tree_dot(grad_vector, (grad_vector**ω - grad_prev**ω).ω)
    denominator = sum_squares(grad_prev)
    # This triggers under two scenarios: (a) at the very start, for which our
    # `grad_prev` is initialised at zero, and (b) at convergence, for which we no longer
    # have a gradient. In either case we set β=0 to revert to just gradient descent.
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    out = jnp.where(pred, jnp.clip(numerator / safe_denom, min=0), 0)
    return cast(Scalar, out)


def fletcher_reeves(grad: Y, grad_prev: Y, y_diff_prev: Y) -> Scalar:
    """The Fletcher--Reeves formula for β. Used with [`optimistix.NonlinearCG`][] and
    [`optimistix.NonlinearCGDescent`][]."""
    del y_diff_prev
    numerator = sum_squares(grad)
    denominator = sum_squares(grad_prev)
    # Triggers at initialisation and convergence, as above.
    pred = denominator > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, 0)


def hestenes_stiefel(grad: Y, grad_prev: Y, y_diff_prev: Y) -> Scalar:
    """The Hestenes--Stiefel formula for β. Used with [`optimistix.NonlinearCG`][] and
    [`optimistix.NonlinearCGDescent`][]."""
    grad_diff = (grad**ω - grad_prev**ω).ω
    numerator = tree_dot(grad, grad_diff)
    denominator = -tree_dot(y_diff_prev, grad_diff)
    # Triggers at initialisation and convergence, as above.
    pred = jnp.abs(denominator) > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, 0)


def dai_yuan(grad: Y, grad_prev: Y, y_diff_prev: Y) -> Scalar:
    """The Dai--Yuan formula for β. Used with [`optimistix.NonlinearCG`][] and
    [`optimistix.NonlinearCGDescent`][]."""
    numerator = sum_squares(grad)
    denominator = -tree_dot(y_diff_prev, (grad**ω - grad_prev**ω).ω)
    # Triggers at initialisation and convergence, as above.
    pred = jnp.abs(denominator) > jnp.finfo(denominator.dtype).eps
    safe_denom = jnp.where(pred, denominator, 1)
    return jnp.where(pred, numerator / safe_denom, 0)


class _NonlinearCGDescentState(eqx.Module, Generic[Y]):
    y_diff: Y
    grad: Y


class NonlinearCGDescent(
    AbstractDescent[
        Y,
        FunctionInfo.EvalGrad
        | FunctionInfo.EvalGradHessian
        | FunctionInfo.EvalGradHessianInv
        | FunctionInfo.ResidualJac,
        _NonlinearCGDescentState,
    ],
):
    """The nonlinear conjugate gradient step."""

    method: Callable[[Y, Y, Y], Scalar]

    def init(
        self,
        y: Y,
        f_info_struct: FunctionInfo.EvalGrad
        | FunctionInfo.EvalGradHessian
        | FunctionInfo.EvalGradHessianInv
        | FunctionInfo.ResidualJac,
    ) -> _NonlinearCGDescentState:
        del f_info_struct
        return _NonlinearCGDescentState(
            y_diff=tree_full_like(y, 0),
            grad=tree_full_like(y, 0),
        )

    def query(
        self,
        y: Y,
        f_info: FunctionInfo.EvalGrad
        | FunctionInfo.EvalGradHessian
        | FunctionInfo.EvalGradHessianInv
        | FunctionInfo.ResidualJac,
        state: _NonlinearCGDescentState,
    ) -> _NonlinearCGDescentState:
        del y
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
                "Cannot use `NonlinearCGDescent` with this solver. This is because "
                "`NonlinearCGDescent` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )
        # On the very first step, we have (from `optim_init`) that `state.y_diff = 0`
        # and `state.grad = 0`. For all methods, this implies that `beta = 0`, so
        # `nonlinear_cg_diretion = neg_grad`, and (as desired) we take just a gradient
        # step on the first step.
        # Furthermore, the same mechanism handles convergence: once
        # `state.{grad, y_diff} = 0`, i.e. our previous step hit a local minima, then
        # on this next step we'll again just use gradient descent, and stop.
        beta = self.method(grad, state.grad, state.y_diff)
        neg_grad = (-(grad**ω)).ω
        nonlinear_cg_direction = (neg_grad**ω + beta * state.y_diff**ω).ω
        # Check if this is a descent direction. Use gradient descent if it isn't.
        y_diff = tree_where(
            tree_dot(grad, nonlinear_cg_direction) < 0,
            nonlinear_cg_direction,
            neg_grad,
        )
        return _NonlinearCGDescentState(y_diff=y_diff, grad=grad)

    def step(
        self, step_size: Scalar, state: _NonlinearCGDescentState
    ) -> tuple[Y, RESULTS]:
        return (step_size * state.y_diff**ω).ω, RESULTS.successful


NonlinearCGDescent.__init__.__doc__ = """**Arguments:**

- `method`: A callable `method(vector, vector_prev, diff_prev)` describing how to
    calculate the beta parameter of nonlinear CG. Nonlinear CG uses the previous search
    direction, scaled by beta, and subtracts the gradient to find the next search 
    direction. This parameter, in the nonlinear case, is the same as the parameter β_n
    described e.g. [on Wikipedia](https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method)
    for the linear case.
    Defaults to `polak_ribiere`. Optimistix includes four built-in methods:
    [`optimistix.polak_ribiere`][], [`optimistix.fletcher_reeves`][],
    [`optimistix.hestenes_stiefel`][], and [`optimistix.dai_yuan`][].
"""


class NonlinearCG(AbstractGradientDescent[Y, Aux]):
    """The nonlinear conjugate gradient method.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NonlinearCGDescent[Y]
    search: AbstractSearch[Y, FunctionInfo.EvalGrad, FunctionInfo.Eval, Any]
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Scalar] = max_norm,
        method: Callable[[Y, Y, Y], Scalar] = polak_ribiere,
        # TODO(raderj): replace the default line search with something better.
        search: AbstractSearch[
            Y, FunctionInfo.EvalGrad, FunctionInfo.Eval, Any
        ] = BacktrackingArmijo(decrease_factor=0.5, slope=0.1),
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NonlinearCGDescent(method=method)
        self.search = search
        self.verbose = verbose


NonlinearCG.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `method`: The function which computes `beta` in `NonlinearCG`. Defaults to
    `polak_ribiere`. Optimistix includes four built-in methods:
    [`optimistix.polak_ribiere`][], [`optimistix.fletcher_reeves`][],
    [`optimistix.hestenes_stiefel`][], and [`optimistix.dai_yuan`][], but any
    function `(Y, Y, Y) -> Scalar` will work.
- `search`: The (line) search to use at each step.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `loss`, `step_size` and `y`. For example 
    `verbose=frozenset({"loss"})`.
"""
