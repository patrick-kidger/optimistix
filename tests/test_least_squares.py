import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import pytest

import optimistix as optx

from .helpers import (
    finite_difference_jvp,
    least_squares_fn_minima_init_args,
    least_squares_optimisers,
    simple_nn,
    tree_allclose,
)


smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", least_squares_optimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", least_squares_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_least_squares(solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-4
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    optx_argmin = optx.least_squares(
        fn,
        solver,
        init,
        has_aux=has_aux,
        args=args,
        max_steps=10_000,
        throw=False,
    ).value
    out = fn(optx_argmin, args)
    if has_aux:
        residual, _ = out
    else:
        residual = out
    optx_min = jtu.tree_reduce(
        lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x**2), residual)
    )
    assert tree_allclose(optx_min, minimum, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", least_squares_optimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", least_squares_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_least_squares_jvp(getkey, solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-2
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def least_squares(x, dynamic_args):
        args = eqx.combine(dynamic_args, static_args)

        if _fn == simple_nn:
            adjoint = optx.ImplicitAdjoint(lx.AutoLinearSolver(well_posed=False))
        else:
            adjoint = optx.ImplicitAdjoint()

        return optx.least_squares(
            fn,
            solver,
            x,
            has_aux=has_aux,
            args=args,
            adjoint=adjoint,
            max_steps=10_000,
            throw=False,
        ).value

    optx_argmin = least_squares(init, args)
    expected_out, t_expected_out = finite_difference_jvp(
        least_squares, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    out, t_out = eqx.filter_jvp(
        least_squares, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    assert tree_allclose(out, expected_out, atol=atol, rtol=rtol)
