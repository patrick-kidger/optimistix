import random

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
import pytest
from equinox.internal import ω

from .helpers import (
    finite_difference_jvp,
    fixed_point_fn_init_args,
    PiggybackAdjoint,
    tree_allclose,
)


atol = rtol = 1e-6
_root_finders = (
    optx.Newton(rtol, atol),
    optx.Chord(rtol, atol),
)
smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
def test_root_find(solver, _fn, init, args):
    atol = rtol = 1e-5
    has_aux = random.choice([True, False])

    def root_find_problem(y, args):
        f_val = _fn(y, args)
        return (f_val**ω - y**ω).ω

    if has_aux:
        fn = lambda x, args: (root_find_problem(x, args), smoke_aux)
    else:
        fn = root_find_problem
    optx_root = optx.root_find(
        fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
    ).value
    out = fn(optx_root, args)
    if has_aux:
        fn_val, _ = out
    else:
        fn_val = out
    zeros = jtu.tree_map(jnp.zeros_like, fn_val)
    assert tree_allclose(fn_val, zeros, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
def test_root_find_jvp(getkey, solver, _fn, init, args):
    atol = rtol = 1e-3
    has_aux = random.choice([True, False])

    def root_find_problem(y, args):
        f_val = _fn(y, args)
        return (f_val**ω - y**ω).ω

    if has_aux:
        fn = lambda x, args: (root_find_problem(x, args), smoke_aux)
    else:
        fn = root_find_problem
    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def root_find(x, dynamic_args, *, adjoint):
        args = eqx.combine(dynamic_args, static_args)
        return optx.root_find(
            fn,
            solver,
            x,
            has_aux=has_aux,
            args=args,
            max_steps=10_000,
            adjoint=adjoint,
            throw=False,
        ).value

    otd = optx.ImplicitAdjoint()
    expected_out, t_expected_out = finite_difference_jvp(
        root_find,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=otd,
    )
    out, t_out = eqx.filter_jvp(
        root_find,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=otd,
    )
    dto = PiggybackAdjoint()
    expected_out2, t_expected_out2 = finite_difference_jvp(
        root_find,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=dto,
    )
    out2, t_out2 = eqx.filter_jvp(
        root_find,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=dto,
    )
    assert tree_allclose(expected_out2, expected_out, atol=atol, rtol=rtol)
    assert tree_allclose(out, expected_out, atol=atol, rtol=rtol)
    assert tree_allclose(out2, expected_out, atol=atol, rtol=rtol)
    assert tree_allclose(t_expected_out2, t_expected_out, atol=atol, rtol=rtol)
    assert tree_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)
    assert tree_allclose(t_out2, t_expected_out, atol=atol, rtol=rtol)


def test_bisection_flip():
    for fn in (lambda x, _: x, lambda x, _: -x):
        options = dict(lower=-1, upper=2)
        sol = optx.root_find(
            fn, optx.Bisection(rtol=1e-4, atol=1e-4), 1.0, options=options
        )
        assert jnp.allclose(0, sol.value, atol=1e-3)


@pytest.mark.parametrize(
    "solver", [optx.Newton(rtol=1e-5, atol=1e-5), optx.Chord(rtol=1e-5, atol=1e-5)]
)
def test_newton_bounded(solver):
    y0 = (jnp.array(0.0), jnp.arange(4.0).reshape(2, 2))

    def f(y, _):
        ya, ((yb, yc), (yd, ye)) = y
        foo = ya + yb + yc + yd + ye
        bar = ya - yb + yc - yd + ye
        return foo * bar, ya, yc**2, jnp.tanh(yd), ye + 1

    # Find roots:
    # Clearly ya=0 yc=0 yd=0, ye=-1.
    # Then foo*bar=0 => foo=0 or bar=0.
    # foo=0 => yb=1
    # bar=0 => yb=-1

    if isinstance(solver, optx.Newton):
        tol = 1e-4
    else:
        tol = 1e-2

    lower_bound = (-jnp.inf, jnp.array([[0, -jnp.inf], [-jnp.inf, -jnp.inf]]))
    true_lower_root = (jnp.array(0.0), jnp.array([[1.0, 0.0], [0.0, -1.0]]))
    y0 = (jnp.array(0.1), jnp.array([[1.1, 0.1], [0.1, -1.1]]))
    lower_root = optx.root_find(f, solver, y0, options=dict(lower=lower_bound)).value
    assert tree_allclose(lower_root, true_lower_root, rtol=tol, atol=tol)

    upper_bound = (jnp.inf, jnp.array([[0, jnp.inf], [jnp.inf, jnp.inf]]))
    true_upper_root = (jnp.array(0.0), jnp.array([[-1.0, 0.0], [0.0, -1.0]]))
    y0 = (jnp.array(0.1), jnp.array([[-1.1, 0.1], [0.1, -1.1]]))
    upper_root = optx.root_find(f, solver, y0, options=dict(upper=upper_bound)).value
    assert tree_allclose(upper_root, true_upper_root, rtol=tol, atol=tol)


def test_root_via_min():
    def f(y, _):
        ya, (yb, yc) = y
        return jnp.tanh(ya + 0.1), jnp.tanh(yb - 0.5), jnp.tanh(yc * 2)

    y0 = jnp.array(0.5), jnp.array([-0.3, 0.7])
    sol = optx.root_find(f, optx.BFGS(rtol=1e-8, atol=1e-8), y0)
    assert tree_allclose(sol.value, (jnp.array(-0.1), jnp.array([0.5, 0.0])))


def test_bad_root_via_min():
    def f(y, _):
        ya, (yb, yc) = y
        return 1.0, jnp.tanh(yb - 0.5), jnp.tanh(yc * 2)

    y0 = jnp.array(0.5), jnp.array([-0.3, 0.7])
    sol = optx.root_find(f, optx.BFGS(rtol=1e-8, atol=1e-8), y0, throw=False)
    assert sol.result == optx.RESULTS.nonlinear_max_steps_reached


@pytest.mark.parametrize("solver_cls", (optx.Newton, optx.Chord))
def test_newton_chord_small_diff(solver_cls):
    def f(y, _):
        return y

    solver = solver_cls(rtol=1e-5, atol=1e-5, cauchy_termination=False)
    sol = optx.root_find(f, solver, 0.0, throw=False)
    assert sol.result == optx.RESULTS.successful
