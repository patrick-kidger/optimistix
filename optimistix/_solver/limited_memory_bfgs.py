from collections.abc import Callable
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar

from .._custom_types import Aux, Y
from .._misc import (
    filter_cond,
    max_norm,
    tree_dot,
)
from .._search import (
    FunctionInfo,
)
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent
from .quasi_newton import AbstractQuasiNewton, AbstractQuasiNewtonUpdate


_Hessian = TypeVar(
    "_Hessian", FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv
)


class _LBFGSInverseHessianUpdateState(eqx.Module, Generic[Y]):
    """
    State variables for Algorithm 4.1 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    **Arguments**:

    - `index_start`: Index of the most recent update in the circular buffer.
    - `y_diff_history`: History of parameter updates `s_k = x_{k+1} - x_k`, in the
        notation od the paper.
    - `grad_diff_history`: History of gradient updates `y_k = g_{k+1} - g_k`, in the
        notation od the paper.
    - `inner_history`: Reciprocal dot products `inner_history = 1 / ⟨s_k, y_k⟩`, in the
        notation of the paper.
    """

    index_start: Scalar
    history_length: int
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    inner_history: Array


class _LBFGSHessianUpdateState(eqx.Module, Generic[Y]):
    r"""
    State variables for Algorithm 3.2 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    This holds a ring buffer of the history of differences in the optimisation variable
    `y` and the gradients `grad`, at the last `n` accepted steps, where `n` is the
    history length.

    **Arguments:**

    - `index_start`: Index of the most recent update in the circular buffer.
    - `y_diff_history`: Circular buffer containing the history of differences in the
        optimisation variable `y` between consecutive accepted steps. In most textbooks
        and in the paper above, the difference in the optimisation variable at iteration
        `k` is denoted as `s_k` and refers to $y_{k+1} - y_k$`. We use the term `y_diff`
        here to maintain consistency with variable names in Optimistix.
        The oldest element is at index `index_start % history_len`.
    - `grad_diff_history`: Circular buffer containing the history of differences in the
        gradient values. Similarly to `y_diff_history`, we follow the Optimistix
        nomenclature rather than the textbook one, where the difference in the gradients
        is usually denoted as `y_k`.
        Indexation is handled as for `y_diff_history`.
    - `y_diff_grad_diff_cross_inner`: Lower triangular matrix with the inner products
        between parameters and gradient difference histories. This parameter corresponds
        to `L_k` in the paper, see def. (2.18), with

            L[ij] = y_diff[i-1]^T grad_diff[j-1] if i > j, 0 otherwise,

        for each iteration `k` that is part of the history (k omitted for clarity).
    - `y_diff_grad_diff_inner`: Array containing the inner products of the differences
        in `y` and the gradients. This parameter corresponds to `diag(D_k)` from the
        paper, see def. (2.7). `[D_k]_{ii} = s_{i-1}^T \cdot y_{i-1}`.
    - `y_diff_cross_inner`: outer product of the parameter difference history. In the
        paper notation, this is equal to `S_{k-1}^T \cdot S_{k-1}`, which is a matrix of
        shape `(history_length, history_length)`.
    """

    index_start: Scalar
    history_length: int
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    y_diff_grad_diff_cross_inner: Float[Array, " history_length history_length"]
    y_diff_grad_diff_inner: Float[Array, " history_length"]
    y_diff_cross_inner: Float[Array, " history_length history_length"]


_LBFGSUpdateState = TypeVar(
    "_LBFGSUpdateState", _LBFGSInverseHessianUpdateState, _LBFGSHessianUpdateState
)


v_tree_dot = jax.vmap(tree_dot, in_axes=(0, None), out_axes=0)


def _lbfgs_inverse_hessian_operator_fn(
    grad: Y,
    state: _LBFGSInverseHessianUpdateState,
) -> Y:
    """
    LBFGS descent linear operator. Implements the L-BFGS update as in Algorithm 4.1 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    **Arguments:**

    - `grad`: The gradient at the current iterate, which is used to compute the descent
        direction.
    - `state`: The current state of the L-BFGS inverse Hessian update, which contains
        the history of parameter and gradient differences, as well as the inner products
        needed to compute the Hessian approximation.
    """
    circ_index = (
        jnp.arange(state.history_length) + state.index_start
    ) % state.history_length

    # First loop: iterate backwards and compute alpha coefficients. This corresponds to
    # step 2 of Algorithm 4.1 in Byrd et al. 1994.
    def backward_iter(descent_direction, indx):
        y_diff, grad_diff, inner = jtu.tree_map(
            lambda x: x[indx],
            (state.y_diff_history, state.grad_diff_history, state.inner_history),
        )
        alpha = inner * tree_dot(y_diff, descent_direction)
        descent_direction = (descent_direction**ω - alpha * grad_diff**ω).ω
        return descent_direction, alpha

    descent_direction, alpha = jax.lax.scan(
        backward_iter, grad, circ_index, reverse=True
    )

    # Here we choose to initialise the inverse Hessian as a diagonal matrix whose
    # entries are defined by the ratio of the inner products `y_diff * grad_diff` and
    # `grad_diff * grad_diff`. This is a common choice, suggested e.g. in
    # https://en.wikipedia.org/wiki/Limited-memory_BFGS#Algorithm
    # We know that gamma > 0 because we only update the Hessian approximation if the
    # inner product is positive, to maintain positive definiteness of the Hessian
    # approximation, and thus this operator is only ever called in that case.
    latest_y_diff, latest_grad_diff = jtu.tree_map(
        lambda x: x[(state.index_start - 1) % state.history_length],
        (state.y_diff_history, state.grad_diff_history),
    )
    y_grad_diff_inner = tree_dot(latest_y_diff, latest_grad_diff)
    grad_diff_norm_sq = tree_dot(latest_grad_diff, latest_grad_diff)
    pred = grad_diff_norm_sq > 0
    safe_grad_diff_norm_sq = jnp.where(pred, grad_diff_norm_sq, 1.0)
    gamma_k = jnp.where(pred, y_grad_diff_inner / safe_grad_diff_norm_sq, 1.0)
    descent_direction = (gamma_k * descent_direction**ω).ω

    # Second loop: iterate forwards and apply correction using stored alpha.
    # This corresponds to step 4 of Algorithm 4.1 in Byrd et al. 1994.
    def forward_iter(args, indx):
        descent_direction, alpha = args
        current_alpha = alpha[indx]
        inner = state.inner_history[circ_index[indx]]
        y_diff, grad_diff = jtu.tree_map(
            lambda x: x[circ_index[indx]],
            (state.y_diff_history, state.grad_diff_history),
        )
        current_beta = inner * tree_dot(grad_diff, descent_direction)
        descent_direction = (
            descent_direction**ω + (y_diff**ω * (current_alpha - current_beta))
        ).ω
        return (descent_direction, alpha), None

    (descent_direction, _), _ = jax.lax.scan(
        forward_iter,
        (descent_direction, alpha),
        jnp.arange(state.history_length),
        reverse=False,
    )
    return descent_direction


def _lbfgs_hessian_operator_fn(
    proposed_step: Y,
    state: _LBFGSHessianUpdateState,
) -> Y:
    """Compact representation of the L-BFGS Hessian update.

    Implements the L-BFGS update as described in Algorithm 3.2 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    This representation is used in limited-memory BFGS algorithms to efficiently store
    and apply approximate Hessians without forming full matrices.

    **Arguments:**

    - `proposed_step`: The proposed step to be applied to the current iterate. In the
        context of an iterative linear solve, this corresponds to

            operator.mv(proposed_step) = vector, which

        is the linear problem we are solving.
    - `state`: The current state of the L-BFGS update, which contains the history of
        parameter and gradient differences, as well as the inner products needed to
        compute the Hessian approximation.
    """
    # state.index_start has already been incremented by 1 after creating the state - so
    # this does the right thing in the first iteration.
    latest_grad_diff, latest_y_diff = jtu.tree_map(
        lambda x: x[(state.index_start - 1) % state.history_length],
        (state.grad_diff_history, state.y_diff_history),
    )

    curvature = tree_dot(latest_y_diff, latest_grad_diff)
    pred = curvature > 0
    safe_curvature = jnp.where(pred, curvature, 1.0)
    gamma_k = jnp.where(
        pred, tree_dot(latest_grad_diff, latest_grad_diff) / safe_curvature, 1.0
    )

    # computation of eqn (3.22) in Byrd at al. 1994
    # eqn (2.26)
    safe_y_diff_grad_diff_inner = jnp.where(
        state.y_diff_grad_diff_inner == 0, 1, state.y_diff_grad_diff_inner
    )
    J_square = gamma_k * state.y_diff_cross_inner + (
        jnp.dot(
            state.y_diff_grad_diff_cross_inner
            / jnp.expand_dims(safe_y_diff_grad_diff_inner, axis=0),
            state.y_diff_grad_diff_cross_inner.T,
        )
    )
    J = jnp.linalg.cholesky(J_square)
    assert J.shape == (state.history_length, state.history_length)

    # step 5 of algorithm 3.2 of Byrd at al. 1994
    descent = jnp.concatenate(
        (
            v_tree_dot(state.grad_diff_history, proposed_step),
            gamma_k * v_tree_dot(state.y_diff_history, proposed_step),
        ),
        axis=0,
    )
    assert descent.shape == (2 * state.history_length,)

    # step 6 of algorithm 3.2: forward backward solve eqn
    sqrt_diag = jnp.sqrt(safe_y_diff_grad_diff_inner)
    low_tri = jnp.block(
        [
            [
                jnp.diag(sqrt_diag),
                jnp.zeros((state.history_length, state.history_length)),
            ],
            [-state.y_diff_grad_diff_cross_inner / sqrt_diag[jnp.newaxis], J],
        ]
    )
    descent = jax.scipy.linalg.solve_triangular(low_tri, descent, lower=True)
    upper_tri = low_tri.T
    upper_tri = upper_tri.at[: state.history_length].multiply(-1)
    descent = jax.scipy.linalg.solve_triangular(upper_tri, descent, lower=False)

    # step 7 of algorithm (3.2)
    descent_step = (
        gamma_k * proposed_step**ω
        - ω(state.grad_diff_history).call(
            lambda x: jnp.einsum("h,h...->...", descent[: state.history_length], x)
        )
        - ω(state.y_diff_history).call(
            lambda x: gamma_k
            * jnp.einsum("h,h...->...", descent[state.history_length :], x),
        )
    ).ω
    return descent_step


def _batched_tree_zeros_like(y, batch_dimension):
    return jtu.tree_map(lambda y: jnp.zeros((batch_dimension, *y.shape)), y)


class LBFGSUpdate(AbstractQuasiNewtonUpdate[Y, _Hessian, _LBFGSUpdateState]):
    r"""L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) update.

    L-BFGS is a memory-efficient variant of the BFGS algorithm that maintains a
    limited history of past updates to approximate the inverse Hessian implicitly,
    without storing the full matrix.

    See [https://en.wikipedia.org/wiki/Limited-memory_BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS).
    """

    use_inverse: bool
    history_length: int

    def init(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, _LBFGSUpdateState]:
        # We're using pyright: ignore on the returned f_info and state because their
        # types depend on the value of `use_inverse`. See
        # https://github.com/patrick-kidger/optimistix/pull/135#discussion_r2155452558
        if self.use_inverse:
            state = _LBFGSInverseHessianUpdateState(
                index_start=jnp.array(0),
                history_length=self.history_length,
                y_diff_history=_batched_tree_zeros_like(y, self.history_length),
                grad_diff_history=_batched_tree_zeros_like(y, self.history_length),
                inner_history=jnp.zeros(self.history_length),
            )
            operator = lx.FunctionLinearOperator(
                lambda y: _lbfgs_inverse_hessian_operator_fn(y, state),
                jax.eval_shape(lambda: y),
                tags=lx.positive_semidefinite_tag,
            )
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, operator)  # pyright: ignore
            return f_info, state  # pyright: ignore
        else:
            state = _LBFGSHessianUpdateState(
                index_start=jnp.array(0),
                history_length=self.history_length,
                y_diff_history=_batched_tree_zeros_like(y, self.history_length),
                grad_diff_history=_batched_tree_zeros_like(y, self.history_length),
                y_diff_grad_diff_cross_inner=jnp.zeros(
                    (self.history_length, self.history_length)
                ),
                y_diff_grad_diff_inner=jnp.ones(self.history_length),
                y_diff_cross_inner=jnp.eye(self.history_length),
            )
            operator = lx.FunctionLinearOperator(
                lambda y: _lbfgs_hessian_operator_fn(y, state),
                jax.eval_shape(lambda: y),
                tags=lx.positive_semidefinite_tag,
            )
            f_info = FunctionInfo.EvalGradHessian(f, grad, operator)  # pyright: ignore
            return f_info, state  # pyright: ignore

    def _no_update(self, inner, grad_diff, y_diff, f_info, hessian_update_state):
        if isinstance(f_info, FunctionInfo.EvalGradHessianInv):
            operator = f_info.hessian_inv
        else:
            operator = f_info.hessian
        return eqx.filter(operator, eqx.is_array), hessian_update_state

    def _update(self, inner, grad_diff, y_diff, f_info, state):
        updated_y_diff_history = jtu.tree_map(
            lambda x, z: x.at[state.index_start % self.history_length].set(z),
            state.y_diff_history,
            y_diff,
        )
        updated_grad_diff_history = jtu.tree_map(
            lambda x, z: x.at[state.index_start % self.history_length].set(z),
            state.grad_diff_history,
            grad_diff,
        )

        if self.use_inverse:
            # Safety check for when the filter_cond that calls `_update` is converted
            # into an elementwise select (e.g., under vmap without JIT). In that case,
            # the non-taken branch may still be evaluated, so we ensure `inner` is never
            # <= eps to prevent `1.0 / inner` from producing NaNs or Infs.
            inner = jnp.where(inner > jnp.finfo(inner.dtype).eps, inner, 1.0)
            updated_inner_history = state.inner_history.at[
                state.index_start % self.history_length
            ].set(1.0 / jnp.asarray(inner))
            updated_state = _LBFGSInverseHessianUpdateState(
                index_start=state.index_start + 1,
                history_length=self.history_length,
                y_diff_history=updated_y_diff_history,
                grad_diff_history=updated_grad_diff_history,
                inner_history=updated_inner_history,
            )
            operator = lx.FunctionLinearOperator(
                lambda y: _lbfgs_inverse_hessian_operator_fn(y, updated_state),
                jax.eval_shape(lambda: y_diff),
                tags=lx.positive_semidefinite_tag,
            )

            # Only return the dynamic part of the operator, keep jaxpr across iterations
            return eqx.filter(operator, eqx.is_array), updated_state

        else:
            # Here we gradually fill in the lower-triangular matrix of inner products
            # between the history of gradient differences and the current difference in
            # the optimisation variable `y`. This matrix has a zero diagonal, it
            # corresponds to `L_k` in the paper, where it is defined by (2.18).
            # At the start of the optimisation, parts of the lower triangle will still
            # be zero. We catch this before computing the Cholesky factorisation by
            # setting the diagonal to 1.0 in the affected rows, and mapping the solution
            # for these elements to a zero vector.
            # (This happens in the operator function.)
            y_diff_grad_diff_cross_inner = state.y_diff_grad_diff_cross_inner.at[
                state.index_start % self.history_length
            ].set(v_tree_dot(state.grad_diff_history, y_diff))
            y_diff_grad_diff_cross_inner = y_diff_grad_diff_cross_inner.at[
                :, state.index_start % self.history_length
            ].set(0)
            assert y_diff_grad_diff_cross_inner.shape == (
                self.history_length,
                self.history_length,
            )

            # Here we update the history of inner products in the circular buffer.
            y_diff_grad_diff_inner = state.y_diff_grad_diff_inner.at[
                state.index_start % self.history_length
            ].set(
                tree_dot(
                    jtu.tree_map(
                        lambda x: x[state.index_start % self.history_length],
                        updated_grad_diff_history,
                    ),
                    y_diff,
                )
            )
            assert y_diff_grad_diff_inner.shape == (self.history_length,)

            # Update the matrix of inner products of `y_diff` by updating one row and
            # one column per iteration. This matrix has nonzero elements for rows up to
            # k and columns up to k, where k is the current iteration index and may be
            # smaller than the history length. Cholesky factorisation works because this
            # matrix is initialised as an identity, and elements > k are mapped to zero
            # by setting the right-hand-side appropriately in the operator function.
            cross_inner = v_tree_dot(
                updated_y_diff_history,
                jtu.tree_map(
                    lambda x: x[state.index_start % self.history_length],
                    updated_y_diff_history,
                ),
            )
            assert cross_inner.shape == (self.history_length,)
            y_diff_cross_inner = state.y_diff_cross_inner.at[
                state.index_start % self.history_length
            ].set(cross_inner)
            y_diff_cross_inner = y_diff_cross_inner.at[
                :, state.index_start % self.history_length
            ].set(cross_inner)
            assert y_diff_cross_inner.shape == (
                self.history_length,
                self.history_length,
            )

            updated_state = _LBFGSHessianUpdateState(
                index_start=state.index_start + 1,
                history_length=self.history_length,
                y_diff_history=updated_y_diff_history,
                grad_diff_history=updated_grad_diff_history,
                y_diff_grad_diff_cross_inner=y_diff_grad_diff_cross_inner,
                y_diff_grad_diff_inner=y_diff_grad_diff_inner,
                y_diff_cross_inner=y_diff_cross_inner,
            )

            operator = lx.FunctionLinearOperator(
                lambda y: _lbfgs_hessian_operator_fn(y, updated_state),
                jax.eval_shape(lambda: y_diff),
                tags=lx.positive_semidefinite_tag,
            )
            # Only return the dynamic part of the operator, keep jaxpr across iterations
            return eqx.filter(operator, eqx.is_array), updated_state

    def __call__(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        state: _LBFGSUpdateState,
    ) -> tuple[_Hessian, _LBFGSUpdateState]:
        if isinstance(f_info, FunctionInfo.EvalGradHessianInv):
            operator = f_info.hessian_inv
        elif isinstance(f_info, FunctionInfo.EvalGradHessian):
            operator = f_info.hessian
        else:
            raise ValueError(
                "Only FunctionInfo.EvalGradHessian and FunctionInfo.EvalGradHessianInv "
                f"are supported for `f_info`, but got {f_info.__class__.__name__}. "
                "L-BFGS updates require an approximation to the the Hessian or the "
                "inverse Hessian at the last accepted point `y`, but this solver does "
                "not provide one."
            )

        # Compute the inner product of the differences in the iterate and the gradient,
        # and update only if the inner product is positive to maintain positive
        # definiteness of the Hessian approximation.
        y_diff = (y_eval**ω - y**ω).ω
        grad_diff = (f_eval_info.grad**ω - f_info.grad**ω).ω
        inner = tree_dot(y_diff, grad_diff)
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps

        # We have a jaxpr in the FunctionLinearOperator, which needs to be filtered to
        # enable downstream checks for equality.
        static_operator = eqx.filter(operator, eqx.is_array, inverse=True)
        new_dynamic_operator, new_state = filter_cond(
            inner_nonzero,
            self._update,
            self._no_update,
            inner,
            grad_diff,
            y_diff,
            f_info,
            state,
        )
        hessian = eqx.combine(static_operator, new_dynamic_operator)
        if isinstance(f_info, FunctionInfo.EvalGradHessianInv):
            new_f_info = FunctionInfo.EvalGradHessianInv(
                f_eval_info.f, f_eval_info.grad, hessian
            )
        else:
            new_f_info = FunctionInfo.EvalGradHessian(
                f_eval_info.f, f_eval_info.grad, hessian
            )
        return new_f_info, new_state  # pyright: ignore


class LBFGS(AbstractQuasiNewton[Y, Aux, _Hessian]):
    """L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm that approximates the inverse Hessian
    using a limited history of gradient and parameter updates. Unlike full BFGS,
    which stores a dense matrix, L-BFGS maintains a memory-efficient
    representation suitable for large-scale problems.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NewtonDescent
    search: BacktrackingArmijo
    hessian_update: AbstractQuasiNewtonUpdate
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        history_length: int = 10,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent()
        self.search = BacktrackingArmijo()
        self.hessian_update = LBFGSUpdate(
            use_inverse=use_inverse, history_length=history_length
        )
        self.verbose = verbose


LBFGS.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `history_length`: Number of parameter and gradient residuals to retain in the 
    L-BFGS history. Larger values can improve accuracy of the inverse Hessian 
    approximation, while smaller values reduce memory and computation. 
    The default is 10.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example
    `verbose=frozenset({"step_size", "loss"})`.
"""
