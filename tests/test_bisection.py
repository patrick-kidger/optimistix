import random

import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest
from equinox.internal import ω

from .helpers import (
    bisection_fn_init_options_args,
    tree_allclose,
    trivial,
)


atol = rtol = 1e-6
_fp_solvers = (optx.FixedPointIteration(rtol, atol),)
smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})



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
