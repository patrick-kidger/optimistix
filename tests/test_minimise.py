import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

import optimistix as optx

from .helpers import (
    finite_difference_jvp,
    minimisation_fn_minima_init_args,
    minimisers,
    tree_allclose,
)


smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", minimisation_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_minimise(solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-4
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn
    optx_argmin = optx.minimise(
        fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
    ).value
    out = fn(optx_argmin, args)
    if has_aux:
        optx_min, _ = out
    else:
        optx_min = out
    assert tree_allclose(optx_min, minimum, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", minimisation_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_minimise_jvp(getkey, solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-4
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def minimise(x, dynamic_args):
        args = eqx.combine(dynamic_args, static_args)
        return optx.minimise(
            fn, solver, x, has_aux=has_aux, args=args, max_steps=10_000, throw=False
        ).value

    optx_argmin = minimise(init, dynamic_args)
    expected_out, t_expected_out = finite_difference_jvp(
        minimise, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    out, t_out = eqx.filter_jvp(
        minimise, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    assert tree_allclose(out, expected_out, atol=atol, rtol=rtol)
