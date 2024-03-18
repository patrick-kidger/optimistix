import contextlib
import random

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import optimistix as optx
import pytest

from .helpers import (
    diagonal_quadratic_bowl,
    finite_difference_jvp,
    least_squares_fn_minima_init_args,
    least_squares_optimisers,
    rosenbrock,
    simple_nn,
    tree_allclose,
)


smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", least_squares_optimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", least_squares_fn_minima_init_args)
def test_least_squares(solver, _fn, minimum, init, args):
    atol = rtol = 1e-4
    has_aux = random.choice([True, False])
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    if isinstance(solver, optx.OptaxMinimiser):
        context = jax.numpy_dtype_promotion("standard")
    else:
        context = contextlib.nullcontext()
    with context:
        optx_argmin = optx.least_squares(
            fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
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
def test_least_squares_jvp(getkey, solver, _fn, minimum, init, args):
    if _fn in (simple_nn, diagonal_quadratic_bowl):
        # These are ridiculously finickity to get references values for the derivatives
        return
    atol = rtol = 1e-2
    has_aux = random.choice([True, False])
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def least_squares(x, dynamic_args, *, adjoint):
        args = eqx.combine(dynamic_args, static_args)
        if isinstance(solver, optx.OptaxMinimiser):
            context = jax.numpy_dtype_promotion("standard")
        else:
            context = contextlib.nullcontext()
        with context:
            return optx.least_squares(
                fn,
                solver,
                x,
                has_aux=has_aux,
                args=args,
                max_steps=10_000,
                adjoint=adjoint,
                throw=False,
            ).value

    if _fn is simple_nn:
        otd = optx.ImplicitAdjoint(linear_solver=lx.AutoLinearSolver(well_posed=False))
    else:
        otd = optx.ImplicitAdjoint()
    out, t_out = eqx.filter_jvp(
        least_squares,
        (init, dynamic_args),
        (t_init, t_dynamic_args),
        adjoint=otd,
    )
    if _fn is rosenbrock:
        # Finite difference does a bad job on this one, but we can figure it out
        # analytically.
        assert isinstance(args, jax.Array)
        expected_out = jtu.tree_map(lambda x: jnp.full_like(x, args), init)
        t_expected_out = jtu.tree_map(lambda x: jnp.full_like(x, t_dynamic_args), init)
    else:
        expected_out, t_expected_out = finite_difference_jvp(
            least_squares,
            (init, dynamic_args),
            (t_init, t_dynamic_args),
            adjoint=otd,
        )
    # TODO(kidger): reinstate once we can do jvp-of-custom_vjp. Right now this errors
    #     because of the line searches used internally.
    #
    # dto = PiggybackAdjoint()
    # expected_out2, t_expected_out2 = finite_difference_jvp(
    #     least_squares, (init, dynamic_args), (t_init, t_dynamic_args), adjoint=dto,
    # )
    # out2, t_out2 = eqx.filter_jvp(
    #     least_squares, (init, dynamic_args), (t_init, t_dynamic_args), adjoint=dto,
    # )
    assert tree_allclose(out, expected_out, atol=atol, rtol=rtol)
    assert tree_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(expected_out2, expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(out2, expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(t_expected_out2, t_expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(t_out2, t_expected_out, atol=atol, rtol=rtol)


def test_gauss_newton_jacrev():
    @jax.custom_vjp
    def f(y, _):
        return dict(bar=y["foo"] ** 2)

    def f_fwd(y, _):
        return f(y, None), jnp.sign(y["foo"])

    def f_bwd(sign, g):
        return dict(foo=sign * g["bar"]), None

    f.defvjp(f_fwd, f_bwd)

    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    y0 = dict(foo=jnp.arange(3.0))
    out = optx.least_squares(f, solver, y0, options=dict(jac="bwd"), max_steps=512)
    assert tree_allclose(out.value, dict(foo=jnp.zeros(3)), rtol=1e-3, atol=1e-2)

    with pytest.raises(TypeError, match="forward-mode autodiff"):
        optx.least_squares(f, solver, y0, options=dict(jac="fwd"), max_steps=512)
