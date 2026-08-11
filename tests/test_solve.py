import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from .helpers import make_nonreal, norm_sq


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_minimise(dtype):
    @jax.grad
    def f(offset):
        def fn(x, _):
            with jax.numpy_dtype_promotion("standard"):
                return norm_sq(x) + offset

        solver = optx.GradientDescent(learning_rate=0.1, rtol=0.1, atol=0.1)
        y0 = make_nonreal(jnp.array(0.0, dtype=dtype))
        return norm_sq(optx.minimise(fn, solver, y0).value)

    f(jnp.array(0.0))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_least_squares(dtype):
    @jax.grad
    def f(offset):
        def fn(x, _):
            with jax.numpy_dtype_promotion("standard"):
                return jnp.conj(x) + offset

        solver = optx.Dogleg(rtol=0.1, atol=0.1)
        y0 = make_nonreal(jnp.array(0.0, dtype=dtype))
        return norm_sq(optx.least_squares(fn, solver, y0).value)

    f(jnp.array(0.0))


def test_root_find():
    @jax.grad
    def f(offset):
        def fn(x, _):
            return x - offset

        solver = optx.Newton(rtol=0.1, atol=0.1)
        return optx.root_find(fn, solver, 0.0).value

    f(0.0)


def test_fixed_point():
    @jax.grad
    def f(offset):
        def fn(x, _):
            return offset

        solver = optx.FixedPointIteration(rtol=0.1, atol=0.1)
        return optx.fixed_point(fn, solver, 0.0).value

    f(0.0)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_forward_mode(dtype):
    def f(y, _):
        return eqxi.nondifferentiable_backward(y)

    optx.least_squares(
        f,
        optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4),
        make_nonreal(jnp.arange(3.0, dtype=dtype)),
        options=dict(jac="fwd"),
    )


# See https://github.com/patrick-kidger/optimistix/issues/155
def test_mixed_dtype():
    def fn(y, _):
        M = jax.numpy.eye(y.shape[-1], dtype=jnp.float32)
        return M @ y

    y0 = jax.numpy.ones(10, dtype=jax.numpy.float32)
    solver = optx.Newton(rtol=1e-3, atol=1e-3)
    optx.root_find(fn, solver, y0, max_steps=1, throw=False)


def test_nonfinite_input():
    def fn(x, _):
        return x**2

    solver = optx.GradientDescent(learning_rate=0.1, rtol=0.1, atol=0.1)
    sol = optx.minimise(fn, solver, float("nan"), throw=False)
    assert sol.result == optx.RESULTS.nonfinite
