import itertools


import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from optimistix._solver.limited_memory_bfgs import (
    _lbfgs_hessian_operator_fn,
    _lbfgs_inverse_hessian_operator_fn,
    _LBFGSHessianUpdateState,
    _LBFGSInverseHessianUpdateState,
    LBFGSUpdate,
    v_tree_dot,
)

from .helpers import tree_allclose


vv_tree_dot = jax.vmap(v_tree_dot, in_axes=(None, 0), out_axes=0)


@pytest.fixture()
def generate_data(request):
    history_len, start_index, return_inverse_state = request.param
    curr_descent = jax.random.normal(jax.random.PRNGKey(123), shape=(3,))
    grad_diff_history = jax.random.normal(
        jax.random.PRNGKey(124), shape=(history_len, 3)
    )
    # enforce positive curvature
    y_diff_history = jnp.sign(grad_diff_history) * jax.random.uniform(
        jax.random.PRNGKey(125), shape=(history_len, 3)
    )

    start_index = jnp.array(start_index)
    inner_history = 1.0 / (y_diff_history * grad_diff_history).sum(axis=1)
    # two-loop recursion state
    if return_inverse_state:
        return (
            curr_descent,
            grad_diff_history,
            y_diff_history,
            inner_history,
            start_index,
        )
    # compact representation state variables
    else:
        y_diff_cross_inner = vv_tree_dot(y_diff_history, y_diff_history)
        y_diff_grad_diff_cross_inner = jnp.dot(
            y_diff_history, grad_diff_history.T, precision=jax.lax.Precision.HIGHEST
        )
        y_diff_grad_diff_cross_inner = y_diff_grad_diff_cross_inner - jnp.triu(
            y_diff_grad_diff_cross_inner
        )
        y_diff_grad_diff_inner = jnp.einsum(
            "i...,i...->i", y_diff_history, grad_diff_history
        )

        return (
            curr_descent,
            grad_diff_history,
            y_diff_history,
            y_diff_cross_inner,
            y_diff_grad_diff_cross_inner,
            y_diff_grad_diff_inner,
            inner_history,
            start_index,
        )


@pytest.mark.parametrize(
    "generate_data", [*itertools.product([2, 3], [0, 1, 7], [True])], indirect=True
)
def test_against_naive_bfgs_hessian_inverse_update(generate_data):
    """Naive implementation of a BFGS inverse hessian update.

    See eqn 1.2 of,

    Byrd, Richard H., et al. "A stochastic quasi-Newton method for large-scale
    optimization." SIAM Journal on Optimization 26.2 (2016): 1008-1031.

    """
    (
        curr_descent,
        grad_diff_history,
        y_diff_history,
        inner_history,
        start_index,
    ) = generate_data
    history_len = len(inner_history)

    # initialize hess inv history (as a scaled identity)
    hess_inv_hist = (
        jnp.eye(curr_descent.shape[0])
        * (
            y_diff_history[(start_index - 1) % history_len]
            @ grad_diff_history[(start_index - 1) % history_len]
        )
        / (
            grad_diff_history[(start_index - 1) % history_len]
            @ grad_diff_history[(start_index - 1) % history_len]
        )
    )
    for k in range(history_len):
        circ_all = (jnp.arange(history_len) + start_index) % history_len
        idx_circ = circ_all[k]

        # implement recursion (1.2) from the reference paper, see docstrings
        Vk = jnp.eye(hess_inv_hist.shape[0]) - inner_history[idx_circ] * jnp.outer(
            grad_diff_history[idx_circ], y_diff_history[idx_circ]
        )
        hess_inv_hist = Vk.T @ hess_inv_hist @ Vk + inner_history[idx_circ] * jnp.outer(
            y_diff_history[idx_circ], y_diff_history[idx_circ]
        )

    # recreate the hessian using the two-loops based operator
    state = _LBFGSInverseHessianUpdateState(
        y_diff_history=y_diff_history,
        grad_diff_history=grad_diff_history,
        inner_history=inner_history,
        index_start=start_index,
    )
    op = lx.FunctionLinearOperator(
        lambda y: _lbfgs_inverse_hessian_operator_fn(y, state),
        jax.eval_shape(lambda: curr_descent),
        tags=lx.positive_semidefinite_tag,
    )
    assert tree_allclose(op.as_matrix(), hess_inv_hist)


@pytest.mark.parametrize(
    "generate_data", [*itertools.product([2, 3], [0], [False])], indirect=True
)
def test_against_naive_bfgs_hessian_update(generate_data):
    """Naive implementation of a BFGS hessian update.

    See eqn 2.16 of,

    Byrd, Richard H., et al. "A stochastic quasi-Newton method for large-scale
    optimization." SIAM Journal on Optimization 26.2 (2016): 1008-1031.

    """
    (
        curr_descent,
        grad_diff_history,
        y_diff_history,
        y_diff_cross_inner,
        y_diff_grad_diff_cross_inner,
        y_diff_grad_diff_inner,
        inner_history,
        start_index,
    ) = generate_data
    history_len = len(y_diff_grad_diff_cross_inner)

    # initialize hess inv history (as a scaled identity)
    hess_hist = (
        jnp.eye(curr_descent.shape[0])
        * (grad_diff_history[start_index - 1] @ grad_diff_history[start_index - 1])
        / (y_diff_history[start_index - 1] @ grad_diff_history[start_index - 1])
    )

    for k in range(history_len):
        circ_all = (jnp.arange(history_len) + start_index) % history_len
        idx_circ = circ_all[k]

        # implement recursion (2.16) from the reference paper, see docstrings
        # for reference paper
        Bsk = jnp.dot(hess_hist, y_diff_history[idx_circ])
        sk = y_diff_history[idx_circ]
        hess_hist = (
            hess_hist
            - jnp.outer(Bsk, Bsk) / (sk @ hess_hist @ sk)
            + jnp.outer(grad_diff_history[idx_circ], grad_diff_history[idx_circ])
            * inner_history[idx_circ]
        )

    # recreate the hessian using (4.2), see docstrings for reference paper
    state = _LBFGSHessianUpdateState(
        y_diff_history=y_diff_history,
        grad_diff_history=grad_diff_history,
        index_start=start_index,
        y_diff_grad_diff_cross_inner=y_diff_grad_diff_cross_inner,
        y_diff_grad_diff_inner=y_diff_grad_diff_inner,
        y_diff_cross_inner=y_diff_cross_inner,
    )
    op = lx.FunctionLinearOperator(
        lambda y: _lbfgs_hessian_operator_fn(y, state),
        jax.eval_shape(lambda: curr_descent),
        tags=lx.positive_semidefinite_tag,
    )
    assert tree_allclose(op.as_matrix(), hess_hist)


@pytest.mark.parametrize(
    "generate_data", [*itertools.product([2, 3], [0], [False])], indirect=True
)
def test_inverse_vs_direct_hessian_operator(generate_data):
    """Test that combining the operetors results in the identity."""
    (
        curr_descent,
        grad_diff_history,
        y_diff_history,
        y_diff_cross_inner,
        l_history,
        d_history,
        inner_history,
        start_index,
    ) = generate_data

    state_hess_inv = _LBFGSInverseHessianUpdateState(
        y_diff_history=y_diff_history,
        grad_diff_history=grad_diff_history,
        inner_history=inner_history,
        index_start=start_index,
    )
    state_hess = _LBFGSHessianUpdateState(
        y_diff_history=y_diff_history,
        grad_diff_history=grad_diff_history,
        index_start=start_index,
        y_diff_grad_diff_cross_inner=l_history,
        y_diff_grad_diff_inner=d_history,
        y_diff_cross_inner=y_diff_cross_inner,
    )

    op = lx.FunctionLinearOperator(
        lambda y: _lbfgs_hessian_operator_fn(y, state_hess),
        jax.eval_shape(lambda: curr_descent),
        tags=lx.positive_semidefinite_tag,
    )
    op_inv = lx.FunctionLinearOperator(
        lambda y: _lbfgs_inverse_hessian_operator_fn(y, state_hess_inv),
        jax.eval_shape(lambda: curr_descent),
        tags=lx.positive_semidefinite_tag,
    )
    identity = (op @ op_inv).as_matrix()
    assert tree_allclose(identity, jnp.eye(identity.shape[0]))


@pytest.mark.parametrize("extra_history", [1, 9])
@pytest.mark.parametrize(
    "generate_data", [*itertools.product([2, 3], [0], [False])], indirect=True
)
def test_warmup_phase_compact(generate_data, extra_history):
    """Test that warm-up and full-buffer L-BFGS states produce the same Hessian.

    During the early iterations of L-BFGS, the history buffers are only
    partially filled. This test verifies that the resulting approximate
    Hessian from such a "warm-up" state matches the one produced by a
    full-length buffer containing the same data in its first `k` slots
    and padding elsewhere.

    The test constructs two `_LBFGSHessianUpdateState` objects:
    - `state_short_hist` with exactly `k` entries.
    - `state_long_hist` with full buffer size and the same `k` entries
      in the beginning, followed by unused space.

    It then checks that both states produce equivalent linear operators.
    """
    (
        curr_descent,
        grad_diff_history,
        y_diff_history,
        y_diff_cross_inner,
        y_diff_grad_diff_cross_inner,
        y_diff_grad_diff_inner,
        inner_history,
        start_index,
    ) = generate_data
    state_short_hist = _LBFGSHessianUpdateState(
        index_start=jnp.array(0, dtype=int),
        y_diff_history=y_diff_history,
        grad_diff_history=grad_diff_history,
        y_diff_grad_diff_cross_inner=y_diff_grad_diff_cross_inner,
        y_diff_grad_diff_inner=y_diff_grad_diff_inner,
        y_diff_cross_inner=y_diff_cross_inner,
    )

    history_len = len(inner_history)
    update = LBFGSUpdate(history_length=history_len + extra_history, use_inverse=False)

    _, state_init = update.init(curr_descent, jnp.array(0.0), curr_descent)

    assert isinstance(state_init, _LBFGSHessianUpdateState)

    state_long_hist = _LBFGSHessianUpdateState(
        y_diff_history=state_init.y_diff_history.at[:history_len].set(
            state_short_hist.y_diff_history
        ),
        grad_diff_history=state_init.grad_diff_history.at[:history_len].set(
            state_short_hist.grad_diff_history
        ),
        y_diff_grad_diff_cross_inner=state_init.y_diff_grad_diff_cross_inner.at[
            :history_len, :history_len
        ].set(state_short_hist.y_diff_grad_diff_cross_inner),
        y_diff_grad_diff_inner=state_init.y_diff_grad_diff_inner.at[:history_len].set(
            state_short_hist.y_diff_grad_diff_inner
        ),
        y_diff_cross_inner=state_init.y_diff_cross_inner.at[
            :history_len, :history_len
        ].set(state_short_hist.y_diff_cross_inner),
        index_start=jnp.array(history_len, dtype=int),
    )
    # just to be sure that the init is different
    assert (
        state_long_hist.y_diff_grad_diff_cross_inner.shape[0]
        == history_len + extra_history
    )
    assert (
        state_long_hist.y_diff_grad_diff_inner.shape[0] == history_len + extra_history
    )
    assert state_long_hist.y_diff_cross_inner.shape[0] == history_len + extra_history

    op_full = lx.FunctionLinearOperator(
        lambda y: _lbfgs_hessian_operator_fn(y, state_short_hist),
        jax.eval_shape(lambda: curr_descent),
        tags=lx.positive_semidefinite_tag,
    )
    op_warmup = lx.FunctionLinearOperator(
        lambda y: _lbfgs_hessian_operator_fn(y, state_long_hist),
        jax.eval_shape(lambda: curr_descent),
        tags=lx.positive_semidefinite_tag,
    )
    assert jnp.allclose(op_full.as_matrix(), op_warmup.as_matrix())
