import itertools

import jax
import jax.numpy as jnp
import pytest
from optimistix._solver.quasi_newton import _make_lbfgs_operator

from .helpers import tree_allclose


@pytest.fixture()
def generate_data(request):
    history_len, start_index = request.param
    curr_descent = jax.random.normal(jax.random.PRNGKey(123), shape=(3,))
    grad_diff_history = jax.random.normal(
        jax.random.PRNGKey(124), shape=(history_len, 3)
    )
    # enforce positive curvature
    y_diff_history = jnp.sign(grad_diff_history) * jax.random.uniform(
        jax.random.PRNGKey(125), shape=(history_len, 3)
    )
    inner_history = 1.0 / (y_diff_history * grad_diff_history).sum(axis=1)
    start_index = jnp.array(start_index)
    return curr_descent, grad_diff_history, y_diff_history, inner_history, start_index


@pytest.mark.parametrize(
    "generate_data", [*itertools.product([2, 3], [0, 1])], indirect=True
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
        * (y_diff_history[start_index - 1] @ grad_diff_history[start_index - 1])
        / (grad_diff_history[start_index - 1] @ grad_diff_history[start_index - 1])
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
    op = _make_lbfgs_operator(
        True, y_diff_history, grad_diff_history, inner_history, start_index
    )
    assert tree_allclose(op.as_matrix(), hess_inv_hist)


@pytest.mark.parametrize(
    "generate_data", [*itertools.product([2, 3], [0, 1])], indirect=True
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
        inner_history,
        start_index,
    ) = generate_data
    history_len = len(inner_history)

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
    b_history = jnp.sqrt(inner_history)[:, None] * grad_diff_history
    op = _make_lbfgs_operator(
        False, y_diff_history, grad_diff_history, b_history, start_index
    )
    assert tree_allclose(op.as_matrix(), hess_hist)


@pytest.mark.parametrize(
    "generate_data", [*itertools.product([2, 3], [0, 1])], indirect=True
)
def test_inverse_vs_direct_hessian_operator(generate_data):
    """Test that combining the operetors results in the identity.

    """
    (
        curr_descent,
        grad_diff_history,
        y_diff_history,
        inner_history,
        start_index,
    ) = generate_data

    b_history = jnp.sqrt(inner_history)[:, None] * grad_diff_history
    op = _make_lbfgs_operator(
        False, y_diff_history, grad_diff_history, b_history, start_index
    )
    op_inv = _make_lbfgs_operator(
        True, y_diff_history, grad_diff_history, inner_history, start_index
    )
    identity = (op @ op_inv).as_matrix()
    assert tree_allclose(identity, jnp.eye(identity.shape[0]))
