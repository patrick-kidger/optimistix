from typing import cast, ClassVar, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, Scalar, ScalarLike

from .._custom_types import Y
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


# `mode` flags for the line-search state machine.
_BRACKET = 0
_ZOOM = 1


# `f_info`, at the last accepted point, must carry a gradient so that we can form the
# directional derivative `phi'(0)`.
_FnInfo: TypeAlias = (
    FunctionInfo.EvalGrad
    | FunctionInfo.EvalGradHessian
    | FunctionInfo.EvalGradHessianInv
)
# `f_eval_info`, at the trial point, must also carry a gradient so that we can form
# `phi'(alpha)` for the curvature condition.
_FnEvalInfo: TypeAlias = FunctionInfo.EvalGrad


class _WolfeState(eqx.Module):
    mode: Int[Array, ""]
    step_size: Scalar  # the step size that produced the current `y_eval`
    a_prev: Scalar  # previous bracket point (bracketing phase)
    phi_prev: Scalar
    a_lo: Scalar  # zoom interval [a_lo, a_hi]; `a_lo` is the best point so far
    phi_lo: Scalar
    a_hi: Scalar
    first_in_search: Bool[Array, ""]  # True before the first bracketing evaluation
    num_evals: Int[Array, ""]


class WolfeSearch(AbstractSearch[Y, _FnInfo, _FnEvalInfo, _WolfeState]):
    """A line search satisfying the strong Wolfe conditions.

    That is, this finds a step size `alpha` such that, writing
    `phi(alpha) = f(y + alpha p)` for the descent direction `p`,

    - `phi(alpha) <= phi(0) + c1 alpha phi'(0)` (the Armijo / sufficient-decrease
        condition), and
    - `|phi'(alpha)| <= c2 |phi'(0)|` (the strong curvature condition).

    This is the line search most commonly paired with quasi-Newton methods such as
    [`optimistix.BFGS`][] and [`optimistix.LBFGS`][]: the curvature condition keeps the
    step large enough for the Hessian update to remain well-conditioned.

    The search follows the bracketing/zoom algorithm of Nocedal & Wright, "Numerical
    Optimization" (2nd ed.), Algorithms 3.5 and 3.6, using bisection for the zoom
    interpolation. As Optimistix calls [`optimistix.AbstractSearch.step`][] once per
    solver step (the iteration loop belongs to the solver), the two nested loops of the
    textbook algorithm are expressed here as a single per-call state machine, dispatched
    by a `mode` flag.

    This search requires the gradient at each trial point, so it must be used with a
    solver that provides one. (It sets `requires_grad_eval = True`; see
    [`optimistix.AbstractSearch`][].) All of Optimistix's gradient-based minimisers do
    so.
    """

    requires_grad_eval: ClassVar[bool] = True

    c1: ScalarLike = 1e-4
    c2: ScalarLike = 0.9
    step_init: ScalarLike = 1.0
    step_max: ScalarLike = 1e6
    max_evals: int = 30

    def __post_init__(self):
        self.c1 = eqx.error_if(
            self.c1,
            (self.c1 <= 0) | (self.c1 >= self.c2),  # pyright: ignore
            "`WolfeSearch` requires `0 < c1 < c2 < 1`.",
        )
        self.c2 = eqx.error_if(
            self.c2,
            (self.c2 <= self.c1) | (self.c2 >= 1),  # pyright: ignore
            "`WolfeSearch` requires `0 < c1 < c2 < 1`.",
        )
        self.step_init = eqx.error_if(
            self.step_init,
            self.step_init <= 0,  # pyright: ignore
            "`WolfeSearch(step_init=...)` must be strictly greater than 0.",
        )

    def init(self, y: Y, f_info_struct: _FnInfo) -> _WolfeState:
        del y, f_info_struct
        zero = jnp.array(0.0)
        return _WolfeState(
            mode=jnp.array(_BRACKET),
            step_size=jnp.array(self.step_init),
            a_prev=zero,
            phi_prev=zero,
            a_lo=zero,
            phi_lo=zero,
            a_hi=zero,
            first_in_search=jnp.array(True),
            num_evals=jnp.array(0),
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: _WolfeState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _WolfeState]:
        if not isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
            ),
        ):
            raise ValueError(
                "Cannot use `WolfeSearch` with this solver. This is because "
                "`WolfeSearch` requires gradients of the target function, but this "
                "solver does not evaluate such gradients."
            )
        if not isinstance(f_eval_info, FunctionInfo.EvalGrad):
            raise ValueError(
                "Cannot use `WolfeSearch` with this solver. This is because "
                "`WolfeSearch` requires the gradient at each trial point, which this "
                "solver does not provide."
            )

        # Restrict to the 1D line `phi(a) = f(y + a p)`. We never need `p` explicitly:
        # with `y_diff = y_eval - y = alpha p`, the directional derivatives are
        # `phi'(0) = <grad(y), p> = <grad(y), y_diff> / alpha` and likewise at the trial
        # point. `compute_grad_dot(y_diff)` returns the relevant `<grad, y_diff>`.
        y_diff = (y_eval**ω - y**ω).ω
        alpha = state.step_size
        safe_alpha = jnp.where(alpha > 0, alpha, 1.0)
        phi0 = f_info.as_min()
        phi = f_eval_info.as_min()
        dphi0 = f_info.compute_grad_dot(y_diff) / safe_alpha
        dphi = f_eval_info.compute_grad_dot(y_diff) / safe_alpha

        armijo = phi <= phi0 + self.c1 * alpha * dphi0
        curvature = jnp.abs(dphi) <= -self.c2 * dphi0

        # Bracketing transition (Nocedal & Wright Alg. 3.5). The minimum is bracketed by
        # [a_prev, alpha] if sufficient decrease fails or `phi` stopped decreasing; by
        # [alpha, a_prev] if the slope has turned positive; otherwise we grow the step.
        b_zoom_prev_alpha = (~armijo) | (
            (phi >= state.phi_prev) & ~state.first_in_search
        )
        b_accept = curvature & ~b_zoom_prev_alpha
        b_zoom_alpha_prev = (dphi >= 0) & ~b_zoom_prev_alpha & ~b_accept
        b_grow = ~b_zoom_prev_alpha & ~b_accept & ~b_zoom_alpha_prev
        b_a_lo = jnp.where(b_zoom_prev_alpha, state.a_prev, alpha)
        b_phi_lo = jnp.where(b_zoom_prev_alpha, state.phi_prev, phi)
        b_a_hi = jnp.where(b_zoom_prev_alpha, alpha, state.a_prev)
        b_mode = jnp.where(b_grow, _BRACKET, _ZOOM)
        b_next = jnp.where(
            b_grow, jnp.minimum(2.0 * alpha, self.step_max), 0.5 * (b_a_lo + b_a_hi)
        )

        # Zoom transition (Nocedal & Wright Alg. 3.6), by bisection of [a_lo, a_hi].
        z_shrink_hi = (~armijo) | (phi >= state.phi_lo)
        z_accept = ~z_shrink_hi & curvature
        z_flip = ~z_shrink_hi & ~curvature & (dphi * (state.a_hi - state.a_lo) >= 0)
        z_a_hi = jnp.where(
            z_shrink_hi, alpha, jnp.where(z_flip, state.a_lo, state.a_hi)
        )
        z_a_lo = jnp.where(z_shrink_hi, state.a_lo, alpha)
        z_phi_lo = jnp.where(z_shrink_hi, state.phi_lo, phi)
        z_next = 0.5 * (z_a_lo + z_a_hi)

        # Select the active branch by the current mode.
        in_bracket = state.mode == _BRACKET
        accept_ls = jnp.where(in_bracket, b_accept, z_accept)
        cont_mode = jnp.where(in_bracket, b_mode, _ZOOM)
        cont_next = jnp.where(in_bracket, b_next, z_next)
        cont_a_prev = jnp.where(in_bracket, alpha, state.a_prev)
        cont_phi_prev = jnp.where(in_bracket, phi, state.phi_prev)
        cont_a_lo = jnp.where(in_bracket, b_a_lo, z_a_lo)
        cont_phi_lo = jnp.where(in_bracket, b_phi_lo, z_phi_lo)
        cont_a_hi = jnp.where(in_bracket, b_a_hi, z_a_hi)

        num_evals = state.num_evals + 1
        out_of_budget = num_evals >= self.max_evals
        # On the first solver step, `y`/`f_info` are dummy data and we must accept. On
        # budget exhaustion we accept the best point seen so far, leaving the solver's
        # gradient-based termination to judge convergence (a best-effort fallback, as in
        # SciPy). This is benign near an optimum, where `phi'(0) -> 0` makes the strong
        # curvature condition impossible to satisfy exactly.
        accept = first_step | accept_ls | out_of_budget

        # On acceptance a new line search begins along a new descent direction, so we
        # reset to `step_init`. `first_in_search` guards the first bracket comparison.
        step_size = cast(Scalar, jnp.where(accept, self.step_init, cont_next))
        new_state = _WolfeState(
            mode=jnp.where(accept, _BRACKET, cont_mode),
            step_size=step_size,
            a_prev=jnp.where(accept, 0.0, cont_a_prev),
            phi_prev=jnp.where(accept, phi, cont_phi_prev),
            a_lo=jnp.where(accept, 0.0, cont_a_lo),
            phi_lo=jnp.where(accept, phi0, cont_phi_lo),
            a_hi=jnp.where(accept, 0.0, cont_a_hi),
            first_in_search=jnp.where(accept, True, False),
            num_evals=jnp.where(accept, 0, num_evals),
        )
        return step_size, accept, RESULTS.successful, new_state


WolfeSearch.__init__.__doc__ = """**Arguments:**

- `c1`: the Armijo (sufficient decrease) constant. Must satisfy `0 < c1 < c2 < 1`.
- `c2`: the curvature constant. Must satisfy `0 < c1 < c2 < 1`. Smaller values impose a
    stricter curvature condition; `0.9` is the usual choice for quasi-Newton methods and
    `0.1` for nonlinear conjugate gradients.
- `step_init`: the step size tried first along each new descent direction. Must be
    strictly greater than 0.
- `step_max`: an upper bound on the step size during bracketing.
- `max_evals`: the maximum number of trial points evaluated per line search, after
    which the best point seen so far is accepted.
"""
