import pytest
from .helpers import tree_allclose
import jax.numpy as jnp
import jax
from optimistix._solver.quasi_newton import _make_lbfgs_operator


def loop_backward(current_grad, grad_diff_history, y_diff_history, inner_history):
    """Follow the lbfgs update equation assuming history is sorted."""
    descent = current_grad.copy()
    alpha = jnp.zeros(y_diff_history.shape[0])
    for i, pars in enumerate(zip(inner_history[::-1], y_diff_history[::-1], grad_diff_history[::-1])):
        inner_i, y_diff_i, grad_diff_i = pars
        alpha = alpha.at[len(alpha) - i - 1].set(inner_i * jnp.dot(y_diff_i, descent))
        descent = descent - alpha[len(alpha) - i - 1] * grad_diff_i
    return descent, alpha


def loop_forward(descent, alpha, grad_diff_history, y_diff_history, inner_history):
    y_grad_inner = y_diff_history[0].dot(grad_diff_history[0])
    y_y_inner = jnp.sum(grad_diff_history[0] ** 2)
    gamma_k = jnp.where(y_y_inner > 1e-10, y_grad_inner / y_y_inner, 1.0)
    search_dir = gamma_k * descent
    for ri, si, yi, ai in zip(inner_history, y_diff_history, grad_diff_history, alpha):
        bi = ri * yi.dot(search_dir)
        search_dir += si * (ai - bi)
    return search_dir


def loop_based_lbfgs_descent_dir(current_grad, grad_diff_history, y_diff_history):
    """Loop based LBFGS update implementation to test against for arrays."""
    inner_history = 1. / (y_diff_history * grad_diff_history).sum(axis=1)
    q, alp = loop_backward(current_grad, grad_diff_history, y_diff_history, inner_history)
    return loop_forward(q, alp, grad_diff_history, y_diff_history, inner_history)


@pytest.mark.parametrize("history_len", [1, 2, 3])
@pytest.mark.parametrize("start_index", [0, 1])
def test_lbfgs_operator_against_loop_implementation(history_len, start_index):
    # generate some random data
    jax.config.update("jax_enable_x64", True)
    curr_descent = jax.random.normal(jax.random.PRNGKey(123), shape=(3, ))
    grad_diff_history = jax.random.normal(jax.random.PRNGKey(124), shape=(history_len, 3))
    y_diff_history = jax.random.normal(jax.random.PRNGKey(125), shape=(history_len, 3))
    inner_history = 1. / (y_diff_history * grad_diff_history).sum(axis=1)

    # sort the time axis and compute descent with naive implementation
    sort_idx = (jnp.arange(history_len) + start_index) % history_len
    descent = loop_based_lbfgs_descent_dir(curr_descent, grad_diff_history[sort_idx], y_diff_history[sort_idx])

    # define and apply operators
    op = _make_lbfgs_operator(y_diff_history, grad_diff_history, inner_history, jnp.array(start_index))
    assert tree_allclose(op.mv(curr_descent), descent, rtol=1e-12, atol=1e-12)