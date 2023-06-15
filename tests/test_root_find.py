import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from equinox.internal import ω

import optimistix as optx

from .helpers import (
    finite_difference_jvp,
    fixed_point_fn_init_args,
    shaped_allclose,
)


atol = rtol = 1e-6
_root_finders = (
    optx.Newton(rtol, atol),
    optx.Chord(rtol, atol),
)
smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_root_find(solver, _fn, init, args, has_aux):
    atol = rtol = 1e-4

    def root_find_problem(y, args):
        f_val = _fn(y, args)
        return (f_val**ω - y**ω).ω

    if has_aux:
        fn = lambda x, args: (root_find_problem(x, args), smoke_aux)
    else:
        fn = root_find_problem
    optx_fp = optx.root_find(
        fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
    ).value
    out = fn(optx_fp, args)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    zeros = jtu.tree_map(jnp.zeros_like, f_val)
    assert shaped_allclose(f_val, zeros, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_root_find_jvp(getkey, solver, _fn, init, args, has_aux):
    atol = rtol = 1e-4

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

    # Chord struggles to hit the very min, it gets close enough to pass and then
    # continues to decrease at a slow rate without signaling convergence for a
    # while.
    def root_find(x, dynamic_args):
        args = eqx.combine(dynamic_args, static_args)
        return optx.root_find(
            fn, solver, x, has_aux=has_aux, args=args, max_steps=10_000, throw=False
        ).value

    optx_root = root_find(init, dynamic_args)
    expected_out, t_expected_out = finite_difference_jvp(
        root_find, (optx_root, dynamic_args), (t_init, t_dynamic_args)
    )
    out, t_out = eqx.filter_jvp(
        root_find, (optx_root, dynamic_args), (t_init, t_dynamic_args)
    )
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)
