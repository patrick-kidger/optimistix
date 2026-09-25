from collections.abc import Callable
from typing import Any, Generic, Literal

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._least_squares import AbstractLeastSquaresSolver
from .._misc import default_verbose, filter_cond, max_norm, tree_full_like
from .._solution import RESULTS


class _Factor(eqx.Module):
    """What a trust-region step needs from one Jacobian, all of size `n` or smaller.

    `J_h = J diag(d)` is the scaled Jacobian and `J_h = U diag(s) V^T` its thin SVD.
    """

    s: Float[Array, " p"]
    v: Float[Array, "n p"]
    uf: Float[Array, " p"]  # U^T f
    d: Float[Array, " n"]  # the scaling, 1 / scale_inv
    g: Float[Array, " n"]  # J^T f, unscaled


def _factor(
    jac: Float[Array, "m n"],
    residual: Float[Array, " m"],
    scale_inv: Float[Array, " n"],
) -> _Factor:
    # SciPy takes the SVD of `J_h` itself, which builds `U`, an m x n matrix. Instead,
    # a QR of `[J_h | f]` gives `R` and `Q^T f` together: `J_h` and `R` then share their
    # singular values and right singular vectors, and `U^T f = U_R^T Q^T f`. So nothing
    # larger than the Jacobian is ever formed.
    d = 1 / scale_inv
    n = jac.shape[1]
    r = jnp.linalg.qr(
        jnp.concatenate([jac * d[None, :], residual[:, None]], axis=1), mode="r"
    )
    u_r, s, vt = jnp.linalg.svd(r[:, :n], full_matrices=False)
    uf = u_r.T @ r[:, n]
    g = (vt.T @ (s * uf)) / d
    return _Factor(s=s, v=vt.T, uf=uf, d=d, g=g)


def _solve_trust_region(
    n: int,
    m: int,
    uf: Float[Array, " p"],
    s: Float[Array, " p"],
    v: Float[Array, "n p"],
    delta: Scalar,
    initial_alpha: Scalar,
    rtol: float = 0.01,
    max_iter: int = 10,
) -> tuple[Float[Array, " n"], Scalar]:
    """Solve `min |J_h p + f|` subject to `|p| <= delta`, returning `p` and the
    Levenberg--Marquardt parameter `alpha` with `(J_h^T J_h + alpha I) p = -J_h^T f`.

    The method of Moré (1977), from a single SVD of `J_h`: a Gauss--Newton step when it
    fits, else a safeguarded Newton iteration on `alpha` for `|p(alpha)| = delta`. As
    `scipy.optimize._lsq.common.solve_lsq_trust_region`.
    """

    def phi_and_derivative(alpha, suf):
        denom = s**2 + alpha
        p_norm = jnp.linalg.norm(suf / denom)
        return p_norm - delta, -jnp.sum(suf**2 / denom**3) / p_norm

    suf = s * uf
    eps = jnp.finfo(s.dtype).eps
    full_rank = (s[-1] > eps * m * s[0]) if m >= n else jnp.array(False)
    safe_s = jnp.where(full_rank, s, 1)
    gauss_newton = -v @ (uf / safe_s)
    take_gauss_newton = full_rank & (jnp.linalg.norm(gauss_newton) <= delta)

    alpha_upper = jnp.linalg.norm(suf) / delta
    phi, phi_prime = phi_and_derivative(0.0, jnp.where(full_rank, suf, 1))
    alpha_lower = jnp.where(full_rank, -phi / phi_prime, 0.0)
    guess = jnp.maximum(0.001 * alpha_upper, jnp.sqrt(alpha_lower * alpha_upper))
    alpha = jnp.where(
        jnp.invert(full_rank) & (initial_alpha == 0), guess, initial_alpha
    )

    def cond_fun(carry):
        it, _, _, _, done = carry
        return (it < max_iter) & jnp.invert(done)

    def body_fun(carry):
        it, alpha, lower, upper, _ = carry
        outside = (alpha < lower) | (alpha > upper)
        alpha = jnp.where(
            outside, jnp.maximum(0.001 * upper, jnp.sqrt(lower * upper)), alpha
        )
        phi, phi_prime = phi_and_derivative(alpha, suf)
        upper = jnp.where(phi < 0, alpha, upper)
        ratio = phi / phi_prime
        lower = jnp.maximum(lower, alpha - ratio)
        alpha = alpha - (phi + delta) * ratio / delta
        return it + 1, alpha, lower, upper, jnp.abs(phi) < rtol * delta

    init = (jnp.array(0), alpha, alpha_lower, alpha_upper, jnp.array(False))
    _, alpha, _, _, _ = lax.while_loop(cond_fun, body_fun, init)
    p = -v @ (suf / (s**2 + alpha))
    # Put `p` exactly on the boundary, so that it cannot lie outside the trust region.
    p = p * delta / jnp.linalg.norm(p)
    return (
        jnp.where(take_gauss_newton, gauss_newton, p),
        jnp.where(take_gauss_newton, 0.0, alpha),
    )


def _update_radius(
    delta: Scalar,
    actual: Scalar,
    predicted: Scalar,
    step_norm: Scalar,
    bound_hit: Bool[Array, ""],
) -> tuple[Scalar, Scalar]:
    """As `scipy.optimize._lsq.common.update_tr_radius`."""
    positive = predicted > 0
    ratio = jnp.where(
        positive,
        actual / jnp.where(positive, predicted, 1),
        jnp.where((predicted == 0) & (actual == 0), 1.0, 0.0),
    )
    delta = jnp.where(
        ratio < 0.25,
        0.25 * step_norm,
        jnp.where((ratio > 0.75) & bound_hit, 2.0 * delta, delta),
    )
    return delta, ratio


def _jacobian(
    fn: Fn[Y, Out, Aux], y: Y, args: PyTree, jac: Literal["fwd", "bwd"]
) -> tuple[Float[Array, "m n"], Float[Array, " m"], Aux]:
    """The Jacobian as a matrix, the residual it was taken at (flattened), and aux."""
    y_flat, unravel = ravel_pytree(y)

    def flat(_y_flat):
        out, aux = fn(unravel(_y_flat), args)
        residual, _ = ravel_pytree(out)
        return residual, (residual, aux)

    if jac == "fwd":
        matrix, (residual, aux) = jax.jacfwd(flat, has_aux=True)(y_flat)
    elif jac == "bwd":
        matrix, (residual, aux) = jax.jacrev(flat, has_aux=True)(y_flat)
    else:
        raise ValueError("Only `jac='fwd'` or `jac='bwd'` are valid.")
    return matrix, residual, aux


class _TRFState(eqx.Module, Generic[Y, Aux]):
    first_step: Bool[Array, ""]
    y_eval: Y
    residual: Float[Array, " m"]
    cost: Scalar
    aux: Aux
    factor: _Factor
    scale_inv: Float[Array, " n"]
    delta: Scalar
    # The Levenberg--Marquardt parameter of the step that produced `y_eval`.
    alpha: Scalar
    # The step that produced `y_eval`, in scaled and in original units, and the decrease
    # in cost the local quadratic model predicted for it.
    step_h_norm: Scalar
    step_norm: Scalar
    predicted: Scalar
    terminate: Bool[Array, ""]
    result: RESULTS
    num_steps: Int[Array, ""]
    num_accepted_steps: Int[Array, ""]
    num_steps_since_acceptance: Int[Array, ""]


class TRF(AbstractLeastSquaresSolver[Y, Out, Aux, _TRFState]):
    """The trust-region method of SciPy's `least_squares(method="trf")`, without bounds.

    A Gauss--Newton method with a trust region on the *scaled* step
    `|D^{-1} p| <= Delta`, solved exactly by the method of Moré (1977) from one SVD of
    the scaled Jacobian. With `x_scale="jac"`, `D` is updated from the Jacobian's column
    norms, which makes the method invariant to the scaling of the unknowns. The trust
    region grows only when a step reaches its boundary and the local model predicted it
    well, and shrinks to a quarter of the step otherwise; a step whose residuals are not
    finite shrinks it and is retried. Rejected steps reuse the last Jacobian's
    factorisation, so they cost one evaluation of `fn` each.

    This follows `scipy.optimize._lsq.trf.trf_no_bounds` with `tr_solver="exact"`, step
    for step: from the same starting point it evaluates the same trial points, up to
    floating-point differences in the linear algebra. Bounds, and the "reflective" part
    of SciPy's method that handles them, are not supported.

    A good choice for badly-scaled or ill-conditioned problems, where
    [`optimistix.LevenbergMarquardt`][] can shrink its trust region and fail to recover.

    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.

    !!! info

        The Jacobian is materialised as a dense matrix at every accepted step, and
        factorised once. This is appropriate for problems with up to a few thousand
        unknowns, and any number of residuals.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    x_scale: Literal["jac"] | float | PyTree[ArrayLike]
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        x_scale: Literal["jac"] | float | PyTree[ArrayLike] = "jac",
        verbose: bool | Callable[..., None] = False,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.x_scale = x_scale
        self.verbose = default_verbose(verbose)

    def _fixed_scale_inv(self, y: Y) -> Float[Array, " n"]:
        y_flat, _ = ravel_pytree(y)
        if isinstance(self.x_scale, (int, float)):
            scale = jnp.full_like(y_flat, self.x_scale)
        else:
            scale, _ = ravel_pytree(jtu.tree_map(jnp.asarray, self.x_scale))
            scale = jnp.broadcast_to(scale.astype(y_flat.dtype), y_flat.shape)
        return 1 / scale

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _TRFState:
        if isinstance(self.x_scale, str):
            if self.x_scale != "jac":
                raise ValueError(
                    "`x_scale` must be 'jac', a positive float, or a pytree."
                )
        else:
            # built once here, so that a malformed scale fails before the first step
            self._fixed_scale_inv(y)
        y_flat, _ = ravel_pytree(y)
        n = y_flat.size
        m = sum(leaf.size for leaf in jtu.tree_leaves(f_struct))
        p = min(m, n + 1, n)
        zero = jnp.zeros((), y_flat.dtype)
        return _TRFState(
            first_step=jnp.array(True),
            y_eval=y,
            residual=jnp.zeros(m, y_flat.dtype),
            cost=zero,
            aux=tree_full_like(aux_struct, 0),
            factor=_Factor(
                s=jnp.zeros(p, y_flat.dtype),
                v=jnp.zeros((n, p), y_flat.dtype),
                uf=jnp.zeros(p, y_flat.dtype),
                d=jnp.ones(n, y_flat.dtype),
                g=jnp.zeros(n, y_flat.dtype),
            ),
            scale_inv=jnp.ones(n, y_flat.dtype),
            delta=zero,
            alpha=zero,
            step_h_norm=zero,
            step_norm=zero,
            predicted=zero,
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_steps=jnp.array(0),
            num_accepted_steps=jnp.array(0),
            num_steps_since_acceptance=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _TRFState,
        tags: frozenset[object],
    ) -> tuple[Y, _TRFState, Aux]:
        jac = options.get("jac", "fwd")
        first = state.first_step
        y_flat, unravel = ravel_pytree(y)
        eval_flat, _ = ravel_pytree(state.y_eval)
        n, m = y_flat.size, state.residual.size

        # Evaluate the pending trial point, and judge it against what was predicted.
        out_eval, _ = fn(state.y_eval, args)
        residual_eval, _ = ravel_pytree(out_eval)
        finite = jnp.all(jnp.isfinite(residual_eval))
        cost_eval = 0.5 * jnp.sum(residual_eval**2)
        actual = jnp.where(finite, state.cost - cost_eval, -1.0)
        bound_hit = state.step_h_norm > 0.95 * state.delta
        delta_new, ratio = _update_radius(
            state.delta, actual, state.predicted, state.step_h_norm, bound_hit
        )
        # SciPy's `ftol` and `xtol` tests, both with `rtol`.
        ftol_satisfied = (actual < self.rtol * state.cost) & (ratio > 0.25)
        x_norm = jnp.linalg.norm(y_flat)
        xtol_satisfied = state.step_norm < self.rtol * (self.rtol + x_norm)
        stop = jnp.invert(first) & finite & (ftol_satisfied | xtol_satisfied)
        accept = first | (finite & (actual > 0))
        # A non-finite trial shrinks the region and is retried; a terminating one keeps
        # it as it was.
        moved = finite & jnp.invert(stop)
        alpha = jnp.where(moved, state.alpha * state.delta / delta_new, state.alpha)
        delta = jnp.where(
            finite, jnp.where(moved, delta_new, state.delta), 0.25 * state.step_h_norm
        )

        def accepted(_):
            matrix, residual, aux = _jacobian(fn, state.y_eval, args, jac)
            norms = jnp.sqrt(jnp.sum(matrix**2, axis=0))
            if isinstance(self.x_scale, str):
                scale_inv = jnp.where(
                    first,
                    jnp.where(norms == 0, 1, norms),
                    jnp.maximum(state.scale_inv, norms),
                )
            else:
                scale_inv = self._fixed_scale_inv(y)
            factor = _factor(matrix, residual, scale_inv)
            return (
                eval_flat,
                residual,
                0.5 * jnp.sum(residual**2),
                aux,
                factor,
                scale_inv,
            )

        def rejected(_):
            return (
                y_flat,
                state.residual,
                state.cost,
                state.aux,
                state.factor,
                state.scale_inv,
            )

        new_flat, residual, cost, aux, factor, scale_inv = filter_cond(
            accept, accepted, rejected, None
        )
        # SciPy's initial trust region: the size of the starting point, in scaled units.
        initial_delta = jnp.linalg.norm(new_flat * scale_inv)
        delta = jnp.where(
            first, jnp.where(initial_delta == 0, 1.0, initial_delta), delta
        )
        alpha = jnp.where(first, 0.0, alpha)
        # SciPy's `gtol` test, with `atol`, on the gradient at the current point.
        gtol_satisfied = self.norm(factor.g) < self.atol
        start_nonfinite = first & jnp.invert(finite)
        terminate = stop | gtol_satisfied | start_nonfinite
        result = RESULTS.where(start_nonfinite, RESULTS.nonfinite, RESULTS.successful)

        # The next trial point: the trust-region step from here.
        step_h, alpha_trial = _solve_trust_region(
            n, m, factor.uf, factor.s, factor.v, delta, alpha
        )
        js = factor.s * (factor.v.T @ step_h)
        predicted = -(0.5 * jnp.sum(js**2) + jnp.sum((factor.d * factor.g) * step_h))
        step = factor.d * step_h

        self.verbose(
            num_steps=("Step", state.num_steps),
            accepted_steps=("Accepted steps", state.num_accepted_steps),
            steps_since_acceptance=(
                "Steps since acceptance",
                state.num_steps_since_acceptance,
            ),
            loss_this_step=("Loss on this step", cost_eval),
            loss_last_accepted_step=("Loss on the last accepted step", state.cost),
            step_size=("Trust region radius", delta),
            y=("y", state.y_eval),
            y_last_accepted_step=("y on the last accepted step", y),
        )
        new_state = _TRFState(
            first_step=jnp.array(False),
            y_eval=unravel(new_flat + step),
            residual=residual,
            cost=cost,
            aux=aux,
            factor=factor,
            scale_inv=scale_inv,
            delta=delta,
            alpha=alpha_trial,
            step_h_norm=jnp.linalg.norm(step_h),
            step_norm=jnp.linalg.norm(step),
            predicted=predicted,
            terminate=terminate,
            result=result,
            num_steps=state.num_steps + 1,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
            num_steps_since_acceptance=jnp.where(
                accept, 0, state.num_steps_since_acceptance + 1
            ),
        )
        return unravel(new_flat), new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _TRFState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _TRFState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {"num_accepted_steps": state.num_accepted_steps}


TRF.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve. Plays the part of both SciPy's
    `ftol` (stop when the decrease in cost is smaller than `rtol` times the cost, on a
    step the local model predicted well) and `xtol` (stop when the step is smaller than
    `rtol * (rtol + |y|)`, in the two-norm).
- `atol`: Absolute tolerance for terminating the solve. Plays the part of SciPy's
    `gtol`: stop when `norm(J^T f) < atol`.
- `norm`: The norm of the gradient `J^T f` in the `atol` test. Should be any function
    `PyTree -> Scalar`. Defaults to [`optimistix.max_norm`][], which is SciPy's choice.
- `x_scale`: The characteristic scale of each unknown: the trust region is `Delta`
    times this along each coordinate. Either `"jac"` (the default), to update it from
    the inverse column norms of the Jacobian as in Moré (1977), or a positive float, or
    a pytree of the same structure as `y`. SciPy's default for `method="trf"` is `1.0`.
- `verbose`: Whether to print out extra information about how the solve is proceeding.
    Can either be `False` to print out nothing, or `True` to print out all information,
    or (for customisation) a callable `**kwargs -> None`. If provided as a callable then
    each value will be a 2-tuple of `(str, jax.Array)` providing a human-readable name
    and its corresponding value.
"""
