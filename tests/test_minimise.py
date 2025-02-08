import contextlib
import functools as ft
import random

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import optimistix as optx
import pytest

from .helpers import (
    beale,
    bowl,
    finite_difference_jvp,
    forward_only_fn_init_options_expected,
    matyas,
    minimisation_fn_minima_init_args,
    minimisers,
    tree_allclose,
)


smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize(
    "options", (dict(autodiff_mode="fwd"), dict(autodiff_mode="bwd"))
)
@pytest.mark.parametrize("solver", minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", minimisation_fn_minima_init_args)
def test_minimise(solver, _fn, minimum, init, args, options):
    if isinstance(solver, optx.GradientDescent):
        max_steps = 100_000
    else:
        max_steps = 10_000
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
        optx_argmin = optx.minimise(
            fn,
            solver,
            init,
            has_aux=has_aux,
            args=args,
            options=options,
            max_steps=max_steps,
            throw=False,
        ).value
    optx_min = _fn(optx_argmin, args)
    assert tree_allclose(optx_min, minimum, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "options", (dict(autodiff_mode="fwd"), dict(autodiff_mode="bwd"))
)
@pytest.mark.parametrize("solver", minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", minimisation_fn_minima_init_args)
def test_minimise_jvp(getkey, solver, _fn, minimum, init, args, options):
    if isinstance(solver, (optx.GradientDescent, optx.NonlinearCG)):
        max_steps = 100_000
        atol = rtol = 1e-2
    else:
        max_steps = 10_000
        atol = rtol = 1e-3
    has_aux = random.choice([True, False])
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def minimise(x, dynamic_args, *, adjoint):
        args = eqx.combine(dynamic_args, static_args)
        if isinstance(solver, optx.OptaxMinimiser):
            context = jax.numpy_dtype_promotion("standard")
        else:
            context = contextlib.nullcontext()
        with context:
            return optx.minimise(
                fn,
                solver,
                x,
                has_aux=has_aux,
                args=args,
                options=options,
                max_steps=max_steps,
                adjoint=adjoint,
                throw=False,
            ).value

    otd = optx.ImplicitAdjoint()
    out, t_out = eqx.filter_jit(ft.partial(eqx.filter_jvp, minimise))(
        (init, dynamic_args), (t_init, t_dynamic_args), adjoint=otd
    )
    if _fn is bowl:
        # Finite difference is very inaccurate on this problem.
        expected_out = t_expected_out = jtu.tree_map(jnp.zeros_like, init)
    elif _fn in (beale, matyas):
        if isinstance(solver, optx.NonlinearCG):
            eps = 1e-3
            atol = rtol = 1e-2  # finite difference does a really bad job on this one
        else:
            eps = 1e-4
        expected_out, t_expected_out = finite_difference_jvp(
            minimise,
            (init, dynamic_args),
            (t_init, t_dynamic_args),
            adjoint=otd,
            eps=eps,
        )
    else:
        expected_out, t_expected_out = finite_difference_jvp(
            minimise, (init, dynamic_args), (t_init, t_dynamic_args), adjoint=otd
        )
    # TODO(kidger): reinstate once we can do jvp-of-custom_vjp. Right now this errors
    #     because of the line searches used internally.
    #
    # dto = PiggybackAdjoint()
    # expected_out2, t_expected_out2 = finite_difference_jvp(
    #     minimise, (init, dynamic_args), (t_init, t_dynamic_args), adjoint=dto,
    # )
    # out2, t_out2 = eqx.filter_jvp(
    #     minimise, (init, dynamic_args), (t_init, t_dynamic_args), adjoint=dto,
    # )
    assert tree_allclose(out, expected_out, atol=atol, rtol=rtol)
    if not isinstance(solver, optx.NelderMead):
        # Nelder-Mead does such a bad job that the finite-difference gradients are
        # noticeably different.
        assert tree_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(expected_out2, expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(out2, expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(t_expected_out2, t_expected_out, atol=atol, rtol=rtol)
    # assert tree_allclose(t_out2, t_expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "method",
    [optx.polak_ribiere, optx.fletcher_reeves, optx.hestenes_stiefel, optx.dai_yuan],
)
def test_nonlinear_cg_methods(method):
    solver = optx.NonlinearCG(rtol=1e-10, atol=1e-10, method=method)

    def f(y, _):
        A = jnp.array([[2.0, -1.0], [-1.0, 3.0]])
        b = jnp.array([-100.0, 5.0])
        c = jnp.array(100.0)
        return jnp.einsum("ij,i,j", A, y, y) + jnp.dot(b, y) + c

    # Analytic minimum:
    # 0 = df/dyk
    #   = A_kj y_j + A_ik y_i + b_k
    #   = 2 A_kj y_j + b_k            (A is symmetric)
    # => y = -0.5 A^{-1} b
    #      = [[-0.3, 0.1], [0.1, 0.2]] [-100, 5]
    #      = [29.5, 9]
    y0 = jnp.array([2.0, 3.0])
    sol = optx.minimise(f, solver, y0, max_steps=500)
    assert tree_allclose(sol.value, jnp.array([29.5, 9.0]), rtol=1e-5, atol=1e-5)


def test_optax_recompilation():
    optim1 = optax.chain(
        optax.adam(jnp.array(1e-3)),
        optax.scale_by_schedule(optax.piecewise_constant_schedule(1, {200: 0.1})),
    )
    solver1 = optx.OptaxMinimiser(optim1, rtol=1e-3, atol=1e-3)

    optim2 = optax.chain(
        optax.adam(jnp.array(1e-2)),
        optax.scale_by_schedule(optax.piecewise_constant_schedule(1, {200: 0.1})),
    )
    solver2 = optx.OptaxMinimiser(optim2, rtol=1e-3, atol=1e-3)

    num_called = 0

    def f(x, _):
        nonlocal num_called
        num_called += 1
        return x**2

    with jax.numpy_dtype_promotion("standard"):
        optx.minimise(f, solver1, 1.0)
        num_called_so_far = num_called
        optx.minimise(f, solver2, 1.0)
    assert num_called_so_far == num_called


@pytest.mark.parametrize("solver", minimisers)
@pytest.mark.parametrize(
    "fn, y0, options, expected", forward_only_fn_init_options_expected
)
def test_forward_minimisation(fn, y0, options, expected, solver):
    if isinstance(solver, optx.OptaxMinimiser):  # No support for forward option
        return
    else:
        # Many steps because gradient descent takes ridiculously long
        sol = optx.minimise(fn, solver, y0, options=options, max_steps=2**10)
        assert sol.result == optx.RESULTS.successful
        assert tree_allclose(sol.value, expected, atol=1e-4, rtol=1e-4)
