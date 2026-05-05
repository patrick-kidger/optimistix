import contextlib
import random

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
import pytest

from .helpers import (
    finite_difference_jvp,
    fixed_point_fn_init_args,
    PiggybackAdjoint,
    tree_allclose,
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
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_fixed_point_jvp(getkey, solver, _fn, init, dtype, args):
    if dtype == jnp.complex128:
        context = pytest.warns(match="Complex support in Optimistix is a work in")
    else:
        context = contextlib.nullcontext()
    with context:
        args = jtu.tree_map(lambda x: x.astype(dtype), args)
        init = jtu.tree_map(lambda x: x.astype(dtype), init)
        atol = rtol = 1e-3
        has_aux = random.choice([True, False])
        if has_aux:
            fn = lambda x, args: (_fn(x, args), smoke_aux)
        else:
            fn = _fn

        dynamic_args, static_args = eqx.partition(args, eqx.is_array)
        t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape, dtype=dtype), init)
        t_dynamic_args = jtu.tree_map(
            lambda x: jr.normal(getkey(), x.shape, dtype=dtype), dynamic_args
        )

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
