import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Aux, DescentState, Fn, HessianUpdateState, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    filter_cond,
    identity_pytree,
    lin_to_grad,
    max_norm,
    tree_dot,
    tree_full_like,
    verbose_print,
)
from .._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent


# TODO(jhaffner): should this be its own thing? Or would it not be clearer to state each
# of these two explicitly?
_Hessian = TypeVar(
    "_Hessian",
    FunctionInfo.EvalGradHessian,
    FunctionInfo.EvalGradHessianInv,
)


class AbstractQuasiNewtonUpdate(eqx.Module, Generic[Y, _Hessian, HessianUpdateState]):
    """Abstract base class for quasi-Newton updates to the Hessian approximation. An
    update consumes information about the current and preceding step, the gradient
    values at the respective points, and the previous Hessian approximation and returns
    an updated instance of [`optimistix.FunctionInfo.EvalGradHessian`][] or
    [`optimistix.FunctionInfo.EvalGradHessianInv`][], depending on whether we're
    approximating the Hessian or its inverse.
    """

    @abc.abstractmethod
    def __call__(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: HessianUpdateState,
    ) -> tuple[_Hessian, HessianUpdateState]:
        """Called whenever we want to update the Hessian approximation. This is usually
        in the `accepted` branch of the `step` method of an
        [`optimistix.AbstractQuasiNewton`][] minimiser.

        **Arguments:**

        - `y`: the previous accepted iterate
        - `y_eval`: the current (just accepted) iterate
        - `f_info`: The function value, gradient and Hessian approximation at the
        previous accepted iterate `y`
        - `f_eval_info`: The function value and its gradient at the current (just
            accepted) iterate `y_eval`

        **Returns:**

        The updated Hessian (or Hessian-inverse) approximation.
        """

    @abc.abstractmethod
    def init(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, HessianUpdateState]:
        """
        Initialize the Hessian structure and Hessian update state.

        Set up a template structure of the Hessian to be used (with dummy values), as
        well as the state of the update method, which stores a history of gradients for
        limited-memory Hessian approximations.
        """
        pass


class _LBFGSInverseHessianUpdateState(eqx.Module, Generic[Y]):
    """
    State variables for Algorithm 4.1 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    **Arguments**:
        - `index_start`: Index of the most recent update in the circular buffer.
        - `y_diff_history`: History of parameter updates `s_k = x_{k+1} - x_k`,
          in the notation od the paper.
        - `grad_diff_history`: History of gradient updates `y_k = g_{k+1} - g_k`,
          in the notation od the paper.
        - `inner_history`: Reciprocal dot products
          `inner_history = 1 / ⟨s_k, y_k⟩`,
          in the notation od the paper.
    """

    index_start: Array
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    inner_history: PyTree[Array]


class _LBFGSHessianUpdateState(eqx.Module, Generic[Y]):
    r"""
    State variables for Algorithm 3.2 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    This holds a ring buffer of the history of differences in the optimisation variable
    `y` and the gradients `grad`, at the last n accepted steps. # TODO n vs history

    **Arguments:**

    # TODO what is index_start?
    - `index_start`: An integer scalar array (ndim == 0)
        Specifying the index of the circular buffer corresponding to
        the farther away history.
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

            L[ij] = y_diff[i-1}^T grad_diff[j-1] if i > j, 0 otherwise,

        for each iteration `k` that is part of the history (k omitted for clarity
        ).

    - `y_diff_grad_diff_inner`: Array containing the inner products of the .
       This parameter corresponds to `diag(D_k)` from the paper, see def. (2.7).
      `[D_k]_{ii} = s_{i-1}^T \cdot y_{i-1}`.
    - `y_diff_cross_inner`: outer product of the parameter difference history.
       In the paper notation, this is equal to `S_{k-1}^T \cdot S_{k-1}`,
       which is a matrix of shape `(history_length, history_length)`.

    """

    index_start: Array
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    y_diff_grad_diff_cross_inner: Array
    y_diff_grad_diff_inner: Array

    # y_diff_grad_diff_inner: Array,  # TODO: is this the same as inner_history?
    # Names should be consistent across the Hessian and inverse Hessian updates.
    y_diff_cross_inner: Array


_LBFGSUpdateState = TypeVar(
    "_LBFGSUpdateState", _LBFGSInverseHessianUpdateState, _LBFGSHessianUpdateState
)


v_tree_dot = jax.vmap(tree_dot, in_axes=(0, None), out_axes=0)


def _lbfgs_inverse_hessian_operator_fn(
    pytree: Y,  # TODO name this better?
    state: _LBFGSInverseHessianUpdateState,
) -> Y:
    """
    LBFGS descent linear operator.

    """
    history_len = state.inner_history.shape[0]
    circ_index = (jnp.arange(history_len) + state.index_start) % history_len

    # First loop: iterate backwards and compute alpha coefficients
    def backward_iter(descent_direction, indx):
        y_diff, grad_diff, inner = jtu.tree_map(
            lambda x: x[indx],
            (state.y_diff_history, state.grad_diff_history, state.inner_history),
        )
        alpha = inner * tree_dot(y_diff, descent_direction)
        descent_direction = (descent_direction**ω - alpha * grad_diff**ω).ω
        return descent_direction, alpha

    # Second loop: iterate forwards and apply correction using stored alpha
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

    descent_direction, alpha = jax.lax.scan(
        backward_iter, pytree, circ_index, reverse=True
    )
    latest_y_diff, latest_grad_diff = jtu.tree_map(
        lambda x: x[(state.index_start - 1) % history_len],
        (state.y_diff_history, state.grad_diff_history),
    )
    y_grad_diff_inner = tree_dot(latest_y_diff, latest_grad_diff)
    grad_diff_norm_sq = tree_dot(latest_grad_diff, latest_grad_diff)
    gamma_k = jnp.where(
        grad_diff_norm_sq > 0, y_grad_diff_inner / grad_diff_norm_sq, 1.0
    )
    descent_direction = (gamma_k * descent_direction**ω).ω
    (descent_direction, _), _ = jax.lax.scan(
        forward_iter, (descent_direction, alpha), jnp.arange(history_len), reverse=False
    )
    return descent_direction


def _lbfgs_hessian_operator_fn(
    step: Y,  # TODO name this better?
    # y_diff_history: PyTree[Y],
    # grad_diff_history: PyTree[Y],
    state: _LBFGSHessianUpdateState,
) -> Y:
    """Compact representation of the L-BFGS Hessian update.

    Implements the L-BFGS update as described in Algorithm 3.2 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    This representation is used in limited-memory BFGS algorithms to efficiently store
    and apply approximate Hessians without forming full matrices.
    """

    history_len = jtu.tree_leaves(state.y_diff_history)[0].shape[0]

    latest_grad_diff, latest_y_diff = jtu.tree_map(
        lambda x: x[(state.index_start - 1) % history_len],
        (state.grad_diff_history, state.y_diff_history),
    )

    curvature = tree_dot(latest_y_diff, latest_grad_diff)
    gamma_k = jnp.where(
        curvature > 0, tree_dot(latest_grad_diff, latest_grad_diff) / curvature, 1.0
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

    # step 5 of algorithm 3.2 of Byrd at al. 1994
    descent = jnp.hstack(
        [
            v_tree_dot(state.grad_diff_history, step),
            gamma_k * v_tree_dot(state.y_diff_history, step),
        ]
    )

    # step 6 of algorithm 3.2: forward backward solve eqn
    sqrt_diag = jnp.sqrt(safe_y_diff_grad_diff_inner)  # TODO safe or regular?
    low_tri = jnp.block(
        [
            [jnp.diag(sqrt_diag), jnp.zeros((history_len, history_len))],
            [-state.y_diff_grad_diff_cross_inner / sqrt_diag[jnp.newaxis], J],
        ]
    )
    descent = jax.scipy.linalg.solve_triangular(low_tri, descent, lower=True)
    upper_tri = low_tri.T
    upper_tri = upper_tri.at[:history_len].set(-upper_tri[:history_len])
    descent = jax.scipy.linalg.solve_triangular(upper_tri, descent, lower=False)

    # step 7 of algorithm (3.2)
    descent = (
        gamma_k * step**ω
        - jtu.tree_map(
            lambda x: jnp.einsum("h,h...->...", descent[:history_len], x),
            state.grad_diff_history,
        )
        ** ω
        - jtu.tree_map(
            lambda x: gamma_k * jnp.einsum("h,h...->...", descent[history_len:], x),
            state.y_diff_history,
        )
        ** ω
    ).ω
    return descent


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
    history_length: int = 10

    def init(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, _LBFGSUpdateState]:
        if self.use_inverse:
            state = _LBFGSInverseHessianUpdateState(
                index_start=jnp.array(0, dtype=int),
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
            return f_info, state  # pyright: ignore  # TODO
        else:
            state = _LBFGSHessianUpdateState(
                index_start=jnp.array(0, dtype=int),
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
            return f_info, state  # pyright: ignore  # TODO

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

        # Explicitly hit different code paths for the updates to the approximate inverse
        # or regular Hessian. This is to make pyright happy, as it can then verify that
        # the type of the state matches the expected type of the returned FunctionInfo
        # in __call__ below.
        # update conditionally
        if self.use_inverse:
            updated_inner_history = state.inner_history.at[
                state.index_start % self.history_length
            ].set(1.0 / inner)
            updated_state = _LBFGSInverseHessianUpdateState(
                index_start=state.index_start + 1,
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
            # if not self.use_inverse:
            y_diff_grad_diff_cross_inner = state.y_diff_grad_diff_cross_inner.at[
                state.index_start % self.history_length
            ].set(v_tree_dot(state.grad_diff_history, y_diff))
            y_diff_grad_diff_cross_inner = y_diff_grad_diff_cross_inner.at[
                :, state.index_start % self.history_length
            ].set(0)

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

            # compute S_{k} \cdot s_{k-1} and roll to update the S_k^T \cdot S_k
            cross_inner = v_tree_dot(
                updated_y_diff_history,
                jtu.tree_map(
                    lambda x: x[state.index_start % self.history_length],
                    updated_y_diff_history,
                ),
            )

            y_diff_cross_inner = state.y_diff_cross_inner.at[
                state.index_start % self.history_length
            ].set(cross_inner)
            y_diff_cross_inner = y_diff_cross_inner.at[
                :, state.index_start % self.history_length
            ].set(cross_inner)

            updated_state = _LBFGSHessianUpdateState(
                index_start=state.index_start + 1,
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
                "L-BFGS updates require an approximation the the Hessian or its "
                "inverse at the last accepted point `y`, but this solver does not "
                "provide one."
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
        return new_f_info, new_state  # pyright: ignore - TODO pyright still not happy


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


class _AbstractBFGSDFPUpdate(AbstractQuasiNewtonUpdate[Y, _Hessian, None]):
    """Private intermediate class for BFGS/DFP updates."""

    use_inverse: AbstractVar[bool]

    def init(
        self, y: Y, f: Scalar, grad: Y
    ) -> tuple[
        _Hessian,
        None,
    ]:
        identity_operator = identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_operator)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_operator)
        return f_info, None  # pyright: ignore TODO

    def no_update(self, inner, grad_diff, y_diff, f_info):
        if self.use_inverse:
            return f_info.hessian_inv
        else:
            return f_info.hessian

    @abc.abstractmethod
    def update(
        self,
        inner: PyTree,
        grad_diff: PyTree,
        y_diff: PyTree,
        f_info: _Hessian,
    ) -> lx.PyTreeLinearOperator:
        ...

    def __call__(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: None,
    ) -> tuple[_Hessian, None]:
        f_eval = f_eval_info.f
        grad = f_eval_info.grad
        y_diff = (y_eval**ω - y**ω).ω
        grad_diff = (grad**ω - f_info.grad**ω).ω
        inner = tree_dot(grad_diff, y_diff)

        # In particular inner = 0 on the first step (as then state.grad=0), and so for
        # this we jump straight to the line search.
        # Likewise we get inner <= eps on convergence, and so again we make no update
        # to avoid a division by zero.
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps

        hessian = filter_cond(
            inner_nonzero,
            self.update,
            self.no_update,
            inner,
            grad_diff,
            y_diff,
            f_info,
        )
        if self.use_inverse:
            return FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian), None  # pyright: ignore TODO
        else:
            return FunctionInfo.EvalGradHessian(f_eval, grad, hessian), None  # pyright: ignore TODO


class DFPUpdate(_AbstractBFGSDFPUpdate[Y, _Hessian]):
    """DFP (Davidon–Fletcher–Powell) approximate Hessian updates.

    This is a variant of the BFGS update.

    See [https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula](https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula).
    """

    use_inverse: bool

    def update(self, inner, grad_diff, y_diff, f_info):
        if self.use_inverse:
            assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
            hessian_inv = f_info.hessian_inv
            inv_mvp = hessian_inv.mv(grad_diff)
            term1 = (_outer(y_diff, y_diff) ** ω / inner).ω
            term2 = (_outer(inv_mvp, inv_mvp) ** ω / tree_dot(grad_diff, inv_mvp)).ω
            new_hessian_inv = lx.PyTreeLinearOperator(
                (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                output_structure=jax.eval_shape(lambda: grad_diff),
                tags=lx.positive_semidefinite_tag,
            )
            return new_hessian_inv
        else:
            assert isinstance(f_info, FunctionInfo.EvalGradHessian)
            hessian = f_info.hessian
            mvp = hessian.mv(y_diff)
            mvp_inner = tree_dot(y_diff, mvp)
            diff_outer = _outer(grad_diff, grad_diff)
            mvp_outer = _outer(grad_diff, mvp)
            term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
            term2 = ((_outer(mvp, grad_diff) ** ω + mvp_outer**ω) / inner).ω
            new_hessian = lx.PyTreeLinearOperator(
                (hessian.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                output_structure=jax.eval_shape(lambda: grad_diff),
                tags=lx.positive_semidefinite_tag,
            )
            return new_hessian


DFPUpdate.__init__.__doc__ = """**Arguments:**

- `use_inverse`: The DFP algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves).
"""


class BFGSUpdate(_AbstractBFGSDFPUpdate[Y, _Hessian]):
    """BFGS (Broyden–Fletcher–Goldfarb–Shanno) approximate Hessian updates.

    See [https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm).
    """

    use_inverse: bool

    def update(self, inner, grad_diff, y_diff, f_info):
        if self.use_inverse:
            assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
            hessian_inv = f_info.hessian_inv
            # Use Woodbury identity for rank-1 update of approximate Hessian.
            inv_mvp = hessian_inv.mv(grad_diff)
            mvp_inner = tree_dot(grad_diff, inv_mvp)
            diff_outer = _outer(y_diff, y_diff)
            mvp_outer = _outer(y_diff, inv_mvp)
            term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
            term2 = ((_outer(inv_mvp, y_diff) ** ω + mvp_outer**ω) / inner).ω
            new_hessian_inv = lx.PyTreeLinearOperator(
                (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                output_structure=jax.eval_shape(lambda: grad_diff),
                tags=lx.positive_semidefinite_tag,
            )
            return new_hessian_inv
        else:
            assert isinstance(f_info, FunctionInfo.EvalGradHessian)
            hessian = f_info.hessian
            # BFGS update to the operator directly
            mvp = hessian.mv(y_diff)
            term1 = (_outer(grad_diff, grad_diff) ** ω / inner).ω
            term2 = (_outer(mvp, mvp) ** ω / tree_dot(y_diff, mvp)).ω
            new_hessian = lx.PyTreeLinearOperator(
                (hessian.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                output_structure=jax.eval_shape(lambda: grad_diff),
                tags=lx.positive_semidefinite_tag,
            )
            return new_hessian


BFGSUpdate.__init__.__doc__ = """**Arguments:**

- `use_inverse`: The BFGS algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves).
"""


class _QuasiNewtonState(
    eqx.Module,
    Generic[Y, Aux, SearchState, DescentState, _Hessian, HessianUpdateState],
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
    # update state
    hessian_update_state: HessianUpdateState


class AbstractQuasiNewton(
    AbstractMinimiser[Y, Aux, _QuasiNewtonState], Generic[Y, Aux, _Hessian]
):
    """Abstract quasi-Newton minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information. These updates to the Hessian are handled by the `hessian_update`
    attribute, which should be a subclass of `AbstractQuasiNewtonUpdate`, such as
    `BFGSUpdate` or `DFPUpdate`.

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
    hessian_update: AbstractVar[AbstractQuasiNewtonUpdate]
    verbose: AbstractVar[frozenset[str]]

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
        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        f_info, hessian_update_state = self.hessian_update.init(y, f, grad)
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
            hessian_update_state=hessian_update_state,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
        tags: frozenset[object],
    ) -> tuple[Y, _QuasiNewtonState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)

            f_eval_info, hessian_update_state = self.hessian_update(
                y,
                state.y_eval,
                state.f_info,
                FunctionInfo.EvalGrad(f_eval, grad),
                state.hessian_update_state,
            )

            descent_state = self.descent.query(
                state.y_eval,
                f_eval_info,  # pyright: ignore
                descent_state,
            )
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return (
                state.y_eval,
                f_eval_info,
                aux_eval,
                descent_state,
                terminate,
                hessian_update_state,
            )

        def rejected(descent_state):
            return (
                y,
                state.f_info,
                state.aux,
                descent_state,
                jnp.array(False),
                state.hessian_update_state,
            )

        y, f_info, aux, descent_state, terminate, hessian_update_state = filter_cond(
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

        state = _QuasiNewtonState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
            hessian_update_state=hessian_update_state,
        )
        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
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
        state: _QuasiNewtonState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class BFGS(AbstractQuasiNewton[Y, Aux, _Hessian]):
    """BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.

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
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
        self.search = BacktrackingArmijo()
        self.hessian_update = BFGSUpdate(use_inverse=use_inverse)
        self.verbose = verbose


BFGS.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: The BFGS algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves). The
    default is (b), denoted via `use_inverse=True`. Note that this is incompatible with
    line search methods like [`optimistix.ClassicalTrustRegion`][], which use the
    Hessian approximation `B` as part of their own computations.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example
    `verbose=frozenset({"step_size", "loss"})`.
"""


class DFP(AbstractQuasiNewton[Y, Aux, _Hessian]):
    """DFP (Davidon–Fletcher–Powell) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.

    [`optimistix.BFGS`][] is generally preferred, since it is more numerically stable on
    most problems.

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
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
        self.search = BacktrackingArmijo()
        self.hessian_update = DFPUpdate(use_inverse=use_inverse)
        self.verbose = verbose


DFP.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: The DFP algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves). The
    default is (b), denoted via `use_inverse=True`. Note that this is incompatible with
    line search methods like [`optimistix.ClassicalTrustRegion`][], which use the
    Hessian approximation `B` as part of their own computations.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example
    `verbose=frozenset({"step_size", "loss"})`.
"""


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
