import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from .helpers import tree_allclose


# Test functions and a descent direction at the start point.
def _rosenbrock(x, args=None):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _quadratic(x, args=None):
    matrix = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    offset = jnp.array([1.0, -2.0])
    return 0.5 * x @ matrix @ x - offset @ x


_LINE_SEARCH_CASES = [
    (_rosenbrock, jnp.array([-1.2, 1.0])),
    (_quadratic, jnp.array([5.0, 5.0])),
]

_MINIMISE_CASES = [
    (_rosenbrock, jnp.array([-1.2, 1.0]), jnp.array([1.0, 1.0])),
    (_rosenbrock, jnp.full((6,), -1.0), jnp.ones((6,))),
    (_quadratic, jnp.array([5.0, 5.0]), jnp.array([0.52173913, -1.13043478])),
]


def _run_single_line_search(search, fn, y, p):
    """Drive `search` along the fixed direction `p`, exactly as a solver would, and
    return the accepted step size."""
    value_and_grad = jax.value_and_grad(fn)
    f0, g0 = value_and_grad(y)
    f_info = optx.FunctionInfo.EvalGrad(f0, g0)
    f_info_struct = jax.eval_shape(lambda: f_info)
    state = search.init(y, f_info_struct)

    # First solver step: dummy data, accept is forced, returns the initial step size.
    step_size, _, _, state = search.step(jnp.array(True), y, y, f_info, f_info, state)
    alpha = step_size
    for _ in range(40):
        y_eval = y + alpha * p
        f_eval, g_eval = value_and_grad(y_eval)
        f_eval_info = optx.FunctionInfo.EvalGrad(f_eval, g_eval)
        step_size, accept, _, state = search.step(
            jnp.array(False), y, y_eval, f_info, f_eval_info, state
        )
        if bool(accept):
            return float(alpha)  # the step size that produced the accepted `y_eval`
        alpha = step_size
    raise AssertionError("line search did not terminate")


@pytest.mark.parametrize("c1, c2", [(1e-4, 0.9), (1e-3, 0.4), (0.1, 0.5)])
@pytest.mark.parametrize("fn, y", _LINE_SEARCH_CASES)
def test_satisfies_strong_wolfe(fn, y, c1, c2):
    value_and_grad = jax.value_and_grad(fn)
    _, g0 = value_and_grad(y)
    p = -g0  # steepest descent direction
    search = optx.WolfeSearch(c1=c1, c2=c2)
    alpha = _run_single_line_search(search, fn, y, p)

    phi0 = fn(y)
    dphi0 = jnp.vdot(g0, p)
    assert dphi0 < 0
    f_alpha, g_alpha = value_and_grad(y + alpha * p)
    dphi_alpha = jnp.vdot(g_alpha, p)
    # Strong Wolfe: sufficient decrease and curvature.
    assert f_alpha <= phi0 + c1 * alpha * dphi0 + 1e-10
    assert jnp.abs(dphi_alpha) <= -c2 * dphi0 + 1e-10


def test_requires_grad_eval_flag():
    assert optx.WolfeSearch.requires_grad_eval is True
    assert optx.BacktrackingArmijo.requires_grad_eval is False
    assert optx.AbstractSearch.requires_grad_eval is False


@pytest.mark.parametrize("solver_cls", [optx.BFGS, optx.LBFGS])
@pytest.mark.parametrize("use_inverse", [True, False])
@pytest.mark.parametrize("fn, y0, expected", _MINIMISE_CASES)
def test_minimise_with_wolfe(solver_cls, use_inverse, fn, y0, expected):
    solver = solver_cls(
        rtol=1e-9, atol=1e-9, use_inverse=use_inverse, search=optx.WolfeSearch()
    )
    sol = optx.minimise(fn, solver, y0, max_steps=1024, throw=False)
    assert sol.result == optx.RESULTS.successful
    assert tree_allclose(sol.value, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("fn, y0, expected", _MINIMISE_CASES)
def test_wolfe_jit(fn, y0, expected):
    solver = optx.LBFGS(rtol=1e-9, atol=1e-9, search=optx.WolfeSearch())
    run = jax.jit(
        lambda y: optx.minimise(fn, solver, y, max_steps=1024, throw=False).value
    )
    assert tree_allclose(run(y0), expected, rtol=1e-4, atol=1e-4)


def test_backtracking_still_works():
    # The opt-in eager-gradient path must not change the default-search behaviour.
    fn, y0, expected = _MINIMISE_CASES[2]  # quadratic
    armijo = optx.LBFGS(rtol=1e-9, atol=1e-9)
    default = optx.minimise(fn, armijo, y0, max_steps=1024, throw=False)
    assert tree_allclose(default.value, expected, rtol=1e-4, atol=1e-4)
