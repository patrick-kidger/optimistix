import jax.numpy as jnp
import jax.scipy.optimize as jsp_optimize
import pytest

import optimistix as optx

from .helpers import beale, tree_allclose


def _setup():
    def fun(x, arg1, arg2, arg3):
        a, b = x
        return beale((a, b), (arg1, arg2, arg3))

    args = (jnp.array(1.5), jnp.array(2.25), jnp.array(2.625))
    x0 = jnp.array([2.0, 0.0])
    return fun, args, x0


@pytest.mark.parametrize("method", ("bfgs", "BFGS"))
def test_minimize(method):
    fun, args, x0 = _setup()
    result = optx.compat.minimize(fun, x0, args, method=method)
    assert tree_allclose(result.x, jnp.array([3.0, 0.5]))
    assert tree_allclose(fun(result.x, *args), jnp.array(0.0))


def test_errors():
    fun, args, x0 = _setup()
    # remove test-time beartype wrapping
    minimize = optx.compat.minimize.__wrapped__
    with pytest.raises(ValueError):
        minimize(fun, [2.0, 0.0], args, method="bfgs")  # pyright: ignore

    with pytest.raises(ValueError):
        minimize(fun, x0, args, method="foobar")

    with pytest.raises(TypeError):
        minimize(fun, x0, None, method="bfgs")  # pyright: ignore


def test_maxiter():
    fun, args, x0 = _setup()
    out = optx.compat.minimize(fun, x0, args, method="bfgs", options=dict(maxiter=2))
    assert not out.success
    assert out.status == 1


def test_compare():
    fun, args, x0 = _setup()
    jax_out = jsp_optimize.minimize(fun, x0, args, method="bfgs")
    optx_out = optx.compat.minimize(fun, x0, args, method="bfgs")
    assert type(jax_out).__name__ == type(optx_out).__name__
    assert tree_allclose(jax_out.x, optx_out.x)
    assert tree_allclose(jax_out.success, optx_out.success)
    assert tree_allclose(jax_out.status, optx_out.status)
    assert tree_allclose(jax_out.fun, optx_out.fun)
    assert tree_allclose(jax_out.jac, optx_out.jac, atol=1e-5, rtol=1e-5)
    assert tree_allclose(jax_out.hess_inv, optx_out.hess_inv, atol=1e-2, rtol=1e-2)
    # Don't compare number of iterations -- these may different between the two
    # implementations.
