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


def test_residual_jac():
    # We have a `.compute_grad_dot(vec)` method, which computes `grad^T vec`. If you did
    # this naively this would require reverse-mode autodiff.
    # However the structure of our problem is such that `grad = jac^T res`, so that in
    # fact we are computing `res^T jac vec`. This means we can actually compute this
    # quantity via forward-mode autodiff -- this is more efficient + supports something
    # things that reverse-mode doesn't, and so doing all of this is the reason that
    # `.compute_grad_dot` exists.
    #
    # However, that's not what this test is really testing.
    #
    # The above is all fairly simple to implement. What this test is *really* handling
    # is that the above all holds true for complex autodiff as well.
    #
    # And that is where things get really complicated. It turns out that JAX defines its
    # complex autodiff like so:
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#complex-numbers-and-differentiation
    # and notably this means that in the case of a function f=u+iv: C->C, whose gradient
    # we compute using `jax.grad` (i.e. reverse-mode autodiff with real contangent 1)
    # then in fact the gradient is computed as `du/dx - i du/dy`. Yup, no `v` dependence
    # because our cotangent has zero imaginary part. Note that this is
    # (a) NOT THE SAME as the Wirtinger derivative `df/dz = 0.5 (df/dx - i df/dy)`;
    # (b) IS THE SAME as the Wirtinger derivative for holomorphic functions. (As for
    #     such functions, `du/dx - i du/dy = df/dz` by the Cauchy--Riemann equations.)
    #
    # Confusingly, JAX decided to go for a non-Wirtinger extension to nonholomorphic
    # functions. This certainly makes sense from the perspective of complex autodiff as
    # an extension of real autodiff, but it's still arguably a bit surprising.
    # For example, the Wirtinger derivative of `0.5 |z|^2 = 0.5 z conj(z)` is
    # `0.5 conj(z)` (just differentiate wrt `z` ignoring `conj(z)`). However the JAX
    # derivative is `conj(z)`!
    #
    # (Note that we always need to make a choice for how we define complex derivatives
    # on nonholomorphic functions -- unsurprisingly everyone does at least try to make
    # their complex derivative be the holomorphic derivative on holomorphic functions,
    # so that bit at least isn't in doubt. As a reference point, PyTorch defines their
    # derivative as `conj(dz)`, i.e. the conjugate of the Wirtinger derivative --
    # which is *not* the same as the conjugate Wirtinger derivative `dconj(z)`. For
    # example it computes the gradient of `z^2` as `conj(z)`, not `0`:
    # https://github.com/patrick-kidger/optimistix/pull/61#issuecomment-2169200128)

    # First compute these quantities as complex numbers. In this case the `jax.grad`
    # gradient (which we are testing that `compute_grad` matches) is given by
    # `du / dx - i du/dy`. (Note that the quantity we are differentiating is a real
    # output, so `v=0` anyway.)

    def residual1(y1):
        return y1**2

    def compute1(y1):
        r = residual1(y1)
        jac = lx.MatrixLinearOperator(jax.jacfwd(residual1, holomorphic=True)(y1))
        f_info = optx.FunctionInfo.ResidualJac(r, jac)
        return f_info.as_min(), (f_info.compute_grad(), f_info.compute_grad_dot(z))

    y1 = jnp.array([2 + 3j, 4 + 1j])
    z = jnp.array([-1 + 0j, 2 - 5j])
    true_min = 0.5 * jnp.sum(y1**2 * jnp.conj(y1**2))
    (min1, (grad1, grad_dot1)), true_grad1 = jax.value_and_grad(compute1, has_aux=True)(
        y1
    )
    # Convention: put the conjugate on the first argument, as per
    # https://github.com/patrick-kidger/diffrax/pull/454#issuecomment-2210296643
    true_grad_dot1 = jnp.sum(jnp.conj(true_grad1) * z)

    # Next compute the same quantities using just the real implementation.

    def residual2(y2):
        real, imag = y2
        return real**2 - imag**2, 2 * real * imag

    def compute2(y2):
        r = residual2(y2)
        jac = lx.PyTreeLinearOperator(
            jax.jacfwd(residual2)(y2), jax.eval_shape(lambda: y2)
        )
        f_info = optx.FunctionInfo.ResidualJac(r, jac)
        return f_info.as_min(), (
            f_info.compute_grad(),
            f_info.compute_grad_dot((z.real, z.imag)),
        )

    y2 = (y1.real, y1.imag)
    (min2, (grad2, grad_dot2)), true_grad2 = jax.value_and_grad(compute2, has_aux=True)(
        y2
    )
    true_grad2_real, true_grad2_imag = true_grad2
    true_grad_dot2 = jnp.sum(true_grad2_real * z.real + true_grad2_imag * z.imag)

    # Now check consistency.

    assert tree_allclose(min1, min2)
    assert tree_allclose(min1.astype(jnp.complex128), true_min)

    assert tree_allclose(grad2, true_grad2)
    assert tree_allclose(grad1, true_grad1)
    # Note the conjugate! As above the complex derivative is given by `du/dx - i du/dy`,
    # whilst the real derivative is given by `(du/dx, du/dy)`.
    assert tree_allclose((grad1.real, -grad1.imag), grad2)

    assert tree_allclose(grad_dot2, true_grad_dot2)

    # TODO: figure out what is going on here, complex numbers don't seem to be behaving.
    pytest.skip()
    assert tree_allclose(grad_dot1, true_grad_dot1)
    # For context, the complex dot product between two scalars is
    # `(a + bi)^bar . (c + di) = ac + bd + i(ad - bc)`
    # The real dot product is
    # `(a, b) . (c, d) = ac + bd`
    # In general we expect the real part of the complex dot product to agree with the
    # real dot product.
    assert tree_allclose(grad_dot1.real, grad_dot2)
