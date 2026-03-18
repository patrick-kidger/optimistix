"""Exact second-order Newton minimisers.

This module provides:
- [`optimistix.SteihaugCGDescent`][]: truncated CG for trust-region subproblems.
- [`optimistix.AbstractNewtonMinimiser`][]: base class for exact-Hessian minimisers.
- [`optimistix.LineSearchNewton`][]: Newton with Armijo line-search globalisation.
- [`optimistix.TrustNewton`][]: Newton with classical trust-region globalisation.

These differ from quasi-Newton methods (BFGS, L-BFGS) in that they evaluate the
**exact** Hessian via JAX automatic differentiation at each accepted step, using
a `lineax.JacobianLinearOperator` over `jax.grad(f)` tagged as symmetric.
This means the Hessian is never materialised: it is a lazy operator that computes
Hessian-vector products via forward-over-reverse AD on demand.

Solvers such as [`optimistix.NewtonDescent`][] will materialise the operator if
their linear solver requires a dense matrix (e.g. `lineax.Cholesky()`), while
[`optimistix.SteihaugCGDescent`][] uses Hessian-vector products directly and
never needs the full matrix.

For large-scale problems consider [`optimistix.BFGS`][] or [`optimistix.LBFGS`][].

### Comparison with the quasi-Newton solvers in optimistix

| Attribute          | BFGS / L-BFGS                | LineSearchNewton / TrustNewton  |
|--------------------|------------------------------|---------------------------------|
| Hessian            | Approximate (rank-2 update)  | Exact (FunctionLinearOperator)  |
| Cost per HVP       | O(n)                         | O(cost of one grad eval)        |
| Non-convex support | Limited (requires PD approx) | TrustNewton + SteihaugCGDescent |
| Good for           | Large-scale problems         | Small/medium, accurate solution |
"""

from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._linear_solver import TruncatedCG
from .._misc import (
    cauchy_termination,
    default_verbose,
    max_norm,
    tree_full_like,
    two_norm,
)
from .._search import AbstractDescent, AbstractSearch, FunctionInfo
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent
from .levenberg_marquardt import IndirectDampedNewtonDescent
from .newton_chord import _NoAux
from .quasi_newton import _QuasiNewtonState, AbstractNewtonBase
from .trust_region import ClassicalTrustRegion


def _grad_hessian(
    fn: Fn[Y, Scalar, Aux],
    y: Y,
    args: PyTree,
    tags: frozenset[object],
    *,
    autodiff_mode: str = "bwd",
) -> tuple[Y, lx.FunctionLinearOperator]:
    """Return ``(grad, hessian_operator)`` at ``y``."""
    fn_no_aux = _NoAux(fn)
    if autodiff_mode == "bwd":
        grad_fn = lambda _y: jax.grad(fn_no_aux)(_y, args)
    else:
        grad_fn = lambda _y: jax.jacfwd(lambda y2: fn_no_aux(y2, args))(_y)
    grad, hvp_fn = jax.linearize(grad_fn, y)
    hessian = lx.FunctionLinearOperator(
        hvp_fn, jax.eval_shape(lambda: y), frozenset({lx.symmetric_tag}) | tags
    )
    return grad, hessian


# ---------------------------------------------------------------------------
# SteihaugCGDescent
# ---------------------------------------------------------------------------


class _SteihaugCGDescentState(eqx.Module, Generic[Y]):
    f_info: FunctionInfo.EvalGradHessian
    grad_norm: Scalar  # ||g||, used to scale the trust-region radius


class SteihaugCGDescent(
    AbstractDescent[
        Y,
        FunctionInfo.EvalGradHessian,
        _SteihaugCGDescentState,
    ],
):
    """Steihaug-Toint truncated CG for trust-region subproblems.

    Approximately solves the trust-region subproblem

    ```
    min  g^T p + 0.5 p^T H p
    s.t. ||p|| <= Δ
    ```

    using conjugate gradients that terminate early when:

    1. **Negative curvature**: `d^T H d ≤ 0`. The current search direction is
       extended to the trust-region boundary and returned.
    2. **Boundary hit**: the unconstrained CG step would leave the trust region.
       The step is projected back onto the boundary and returned.
    3. **CG convergence**: `||r_j|| < eta_j * ||g||` where
       `eta_j = min(rtol, sqrt(||g||))` is the Eisenstat-Walker forcing sequence.
       The current iterate is returned.

    Hessian-vector products are computed lazily via forward-over-reverse AD
    (`jax.jvp` of the gradient), so the full Hessian is never materialised.
    This makes the per-step cost O(k * cost_of_grad) for k CG iterations rather
    than O(n²).

    Because no Cholesky factorisation is required and negative curvature is
    handled gracefully, this descent is suitable for **non-convex** problems
    where the Hessian may be indefinite.

    Designed for use with [`optimistix.TrustNewton`][].
    """

    max_steps: int | None = None
    rtol: float = 0.5

    def init(
        self,
        y: Y,
        f_info_struct: FunctionInfo.EvalGradHessian,
    ) -> _SteihaugCGDescentState:
        f_info_init = tree_full_like(f_info_struct, 0, allow_static=True)
        return _SteihaugCGDescentState(f_info=f_info_init, grad_norm=jnp.array(0.0))

    def query(
        self,
        y: Y,
        f_info: FunctionInfo.EvalGradHessian,
        state: _SteihaugCGDescentState,
    ) -> _SteihaugCGDescentState:
        del y, state
        return _SteihaugCGDescentState(f_info=f_info, grad_norm=two_norm(f_info.grad))

    def step(
        self, step_size: Scalar, state: _SteihaugCGDescentState
    ) -> tuple[Y, RESULTS]:
        g = state.f_info.grad
        H = state.f_info.hessian
        delta = state.grad_norm * step_size

        # Eisenstat-Walker: tighten the inner CG tolerance as ||g|| → 0.
        # ew_rtol = min(rtol, sqrt(||g||)) gives the adaptive forcing sequence.
        # At large ||g|| this caps at self.rtol (0.5 by default); as the outer
        # iterate converges the inner solve is tightened automatically.
        ew_rtol = jnp.minimum(jnp.array(self.rtol), jnp.sqrt(state.grad_norm))

        # TruncatedCG solves H p = -g.  With a finite delta it handles both
        # early exits (negative curvature and boundary crossing) internally,
        # projecting onto the trust-region sphere and returning the result
        # directly as .value.
        out = lx.linear_solve(
            H,
            jtu.tree_map(jnp.negative, g),
            TruncatedCG(rtol=self.rtol, atol=0.0, max_steps=self.max_steps),
            options={"delta": delta, "rtol": ew_rtol},
            throw=False,
        )
        return out.value, RESULTS.successful


SteihaugCGDescent.__init__.__doc__ = """**Arguments:**

- `max_steps`: Maximum number of CG iterations per outer Newton step. Defaults
    to `None`, which falls through to `TruncatedCG`'s default of `20 * n`.
    Set explicitly to impose a hard cap (e.g. for large problems).
- `rtol`: Upper bound for the Eisenstat-Walker forcing sequence. The inner CG
    terminates when `||r_j|| < eta_j * ||g||` where
    `eta_j = min(rtol, sqrt(||g||))`. At large `||g||` this caps at `rtol`
    (coarse inner solve); as `||g|| → 0` the inner tolerance is tightened
    automatically, matching scipy's trust-ncg behaviour. Defaults to 0.5.
"""


# ---------------------------------------------------------------------------
# AbstractNewtonMinimiser
# ---------------------------------------------------------------------------


class AbstractNewtonMinimiser(
    AbstractNewtonBase[Y, Aux],
    Generic[Y, Aux],
):
    """Abstract base class for exact second-order Newton minimisers.

    Subclasses [`optimistix.AbstractNewtonBase`][].

    Subclasses evaluate the **exact Hessian** at each accepted step using a
    `lineax.JacobianLinearOperator` wrapping `jax.grad(fn)`, tagged as
    symmetric. The Hessian is never materialised: Hessian-vector products are
    computed lazily via forward-over-reverse AD. If the chosen linear solver
    (e.g. `lineax.Cholesky()`) needs a dense matrix it will materialise on
    demand, but [`optimistix.SteihaugCGDescent`][] avoids this entirely.

    Subclasses must provide the following attributes:

    - `rtol: float`
    - `atol: float`
    - `norm: Callable[[PyTree], Scalar]`
    - `descent: AbstractDescent[Y, FunctionInfo.EvalGradHessian, Any]`
    - `search: AbstractSearch[Y, FunctionInfo.EvalGradHessian, FunctionInfo.Eval, Any]`
    - `verbose: Callable[..., None]`

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation
        to compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to
        `"bwd"`.
    """

    descent: AbstractVar[AbstractDescent[Y, FunctionInfo.EvalGradHessian, Any]]
    search: AbstractVar[
        AbstractSearch[Y, FunctionInfo.EvalGradHessian, FunctionInfo.Eval, Any]
    ]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _QuasiNewtonState:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        hessian_struct = eqx.filter_eval_shape(
            lambda: _grad_hessian(fn, y, args, tags, autodiff_mode=autodiff_mode)[1]
        )
        hessian = tree_full_like(hessian_struct, 0, allow_static=True)
        f_info = FunctionInfo.EvalGradHessian(f, grad, hessian)
        f_info_struct = eqx.filter_eval_shape(lambda: f_info)
        return _QuasiNewtonState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
            hessian_update_state=None,
        )

    def _prepare_step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
        tags: frozenset[object],
    ) -> tuple[Scalar, Aux, Callable[..., Any], None]:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, aux_eval = fn(state.y_eval, args)

        def accepted(descent_state):
            grad, hessian = _grad_hessian(
                fn, state.y_eval, args, tags, autodiff_mode=autodiff_mode
            )
            # Swap in residuals from this step while keeping the static jaxpr
            # from state, so filter_cond sees identical treedefs in both branches.
            dynamic = eqx.filter(hessian, eqx.is_array)
            static = eqx.filter(state.f_info.hessian, eqx.is_array, inverse=True)
            hessian = eqx.combine(dynamic, static)

            f_eval_info = FunctionInfo.EvalGradHessian(f_eval, grad, hessian)
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)

            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)
            return (
                state.y_eval,
                f_eval_info,
                aux_eval,
                descent_state,
                terminate,
                None,
            )

        return f_eval, aux_eval, accepted, None


# ---------------------------------------------------------------------------
# LineSearchNewton
# ---------------------------------------------------------------------------


class LineSearchNewton(AbstractNewtonMinimiser[Y, Aux]):
    """Newton minimiser with Armijo backtracking line-search globalisation.

    At each accepted step an exact Hessian-vector-product operator is built via
    `jax.grad` and `lineax.JacobianLinearOperator`, then the Newton system
    `H δ = -g` is solved to obtain the search direction. An Armijo backtracking
    line search then finds an acceptable step length.

    This is a good choice for **convex or near-convex** problems where the
    Hessian is guaranteed to be (or very close to) positive definite. For
    non-convex problems where the Hessian can be indefinite, prefer
    [`optimistix.TrustNewton`][] with [`optimistix.SteihaugCGDescent`][].

    Supports the following `options`:

    - `autodiff_mode`: `"fwd"` or `"bwd"`. Defaults to `"bwd"`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NewtonDescent
    search: BacktrackingArmijo
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=True),
        verbose: bool | Callable[..., None] = False,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent(linear_solver=linear_solver)
        self.search = BacktrackingArmijo()
        self.verbose = default_verbose(verbose)


LineSearchNewton.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to solve `H δ = -g`. Defaults to
    `lineax.AutoLinearSolver(well_posed=True)`, which assumes the Hessian is
    square and non-singular and dispatches to LU. Use
    `lineax.CG(rtol=..., atol=...)` for large problems, or
    [`optimistix.TruncatedCG`][] to handle indefinite Hessians.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Can be `False`, `True`, or a callable `**kwargs -> None`.
"""


# ---------------------------------------------------------------------------
# TrustNewton
# ---------------------------------------------------------------------------

_UNSET = object()


class TrustNewton(AbstractNewtonMinimiser[Y, Aux]):
    """Newton minimiser with classical trust-region globalisation.

    At each accepted step an exact Hessian-vector-product operator is built via
    `jax.grad` and `lineax.JacobianLinearOperator`, then the trust-region
    subproblem is solved. Two descent directions are supported:

    - **`IndirectDampedNewtonDescent`** (default): solves the true trust-region
      subproblem by root-finding for the Levenberg--Marquardt parameter λ such
      that `‖(H + λI)⁻¹g‖ = Δ` (Conn, Gould, Toint §7.3). Handles indefinite
      Hessians correctly by ensuring `H + λI` is positive definite.
    - **`SteihaugCGDescent`**: solves the trust-region subproblem approximately
      via truncated CG. Handles indefinite Hessians gracefully, never
      materialises the Hessian, and is preferred for **large-scale** problems.

    To use `SteihaugCGDescent`, pass `use_steihaug=True`.

    Supports the following `options`:

    - `autodiff_mode`: `"fwd"` or `"bwd"`. Defaults to `"bwd"`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: IndirectDampedNewtonDescent | SteihaugCGDescent
    search: ClassicalTrustRegion
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = _UNSET,  # type: ignore[assignment]
        use_steihaug: bool = False,
        steihaug_max_steps: int | None = None,
        verbose: bool | Callable[..., None] = False,
    ):
        if use_steihaug and linear_solver is not _UNSET:
            raise ValueError(
                "`linear_solver` has no effect when `use_steihaug=True` because "
                "`SteihaugCGDescent` constructs its own `TruncatedCG` solver "
                "internally. Either remove `linear_solver` or set "
                "`use_steihaug=False`."
            )
        if linear_solver is _UNSET:
            linear_solver = lx.AutoLinearSolver(well_posed=True)
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        if use_steihaug:
            self.descent = SteihaugCGDescent(max_steps=steihaug_max_steps)
        else:
            self.descent = IndirectDampedNewtonDescent(linear_solver=linear_solver)
        self.search = ClassicalTrustRegion()
        self.verbose = default_verbose(verbose)


TrustNewton.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used inside `IndirectDampedNewtonDescent`
    when `use_steihaug=False`. Defaults to
    `lineax.AutoLinearSolver(well_posed=True)`, which assumes the (shifted)
    Hessian is square and non-singular and dispatches to LU. Passing
    `linear_solver` together with `use_steihaug=True` is an error:
    `SteihaugCGDescent` constructs its own internal solver and does not use
    this argument.
- `use_steihaug`: If `True`, use [`optimistix.SteihaugCGDescent`][] to solve
    the trust-region subproblem via truncated CG. This handles indefinite
    Hessians and is recommended for non-convex problems.
- `steihaug_max_steps`: Maximum CG iterations per outer Newton step when
    `use_steihaug=True`. Defaults to `None` (`TruncatedCG` uses `20 * n`).
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Can be `False`, `True`, or a callable `**kwargs -> None`.
"""
