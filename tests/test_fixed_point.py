import random

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from equinox.internal import ω

import optimistix as optx

from .helpers import (
    bisection_fn_init_options_args,
    finite_difference_jvp,
    fixed_point_fn_init_args,
    PiggybackAdjoint,
    tree_allclose,
    trivial,
)


atol = rtol = 1e-6
_fp_solvers = (optx.FixedPointIteration(rtol, atol),)
smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", _fp_solvers)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
def test_fixed_point(solver, _fn, init, args):
    atol = rtol = 1e-4
    has_aux = random.choice([True, False])
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn
    optx_fp = optx.fixed_point(
        fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
    ).value
    out = fn(optx_fp, args)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    assert tree_allclose(optx_fp, f_val, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _fp_solvers)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
def test_fixed_point_jvp(getkey, solver, _fn, init, args):
    atol = rtol = 1e-3
    has_aux = random.choice([True, False])
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def fixed_point(x, dynamic_args, *, adjoint):
        args = eqx.combine(dynamic_args, static_args)
        return optx.fixed_point(
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
        fixed_point,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=otd,
    )
    out, t_out = eqx.filter_jvp(
        fixed_point,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=otd,
    )
    dto = PiggybackAdjoint()
    expected_out2, t_expected_out2 = finite_difference_jvp(
        fixed_point,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=dto,
    )
    out2, t_out2 = eqx.filter_jvp(
        fixed_point,
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


@pytest.mark.parametrize(
    "_fn, init, bisection_options, args", bisection_fn_init_options_args
)
def test_bisection(_fn, init, bisection_options, args):
    solver = optx.Bisection(rtol=1e-6, atol=1e-6)
    atol = rtol = 1e-4
    has_aux = random.choice([True, False])

    def root_find_problem(y, args):
        f_val = _fn(y, args)
        return (f_val**ω - y**ω).ω

    if _fn == trivial:
        bisection_problem = _fn
    else:
        bisection_problem = root_find_problem
    if has_aux:
        fn = lambda x, args: (bisection_problem(x, args), smoke_aux)
    else:
        fn = bisection_problem

    optx_fp = optx.root_find(
        fn,
        solver,
        init,
        has_aux=has_aux,
        args=args,
        options=bisection_options,
        max_steps=10_000,
        throw=False,
    ).value
    out = fn(optx_fp, args)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    zeros = jtu.tree_map(jnp.zeros_like, f_val)
    assert tree_allclose(f_val, zeros, atol=atol, rtol=rtol)
