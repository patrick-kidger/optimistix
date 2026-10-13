import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
import pytest


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("slope", [-1, 1])
@pytest.mark.parametrize("root", [0, 1])
def test_bisection_root_at_endpoint(dtype, slope, root):
    def fn(y, args):
        del args
        return slope * (y - root)

    sol = optx.root_find(
        fn,
        optx.Bisection(rtol=1e-5, atol=1e-5),
        jnp.array(0.5, dtype=dtype),
        options=dict(lower=0.0, upper=1.0),
    )
    assert sol.result == optx.RESULTS.successful
    assert jnp.allclose(sol.value, root, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bisection_zero_width_root_interval(dtype):
    sol = optx.root_find(
        lambda y, _: y - 2,
        optx.Bisection(rtol=1e-5, atol=1e-5),
        jnp.array(2.0, dtype=dtype),
        options=dict(lower=2.0, upper=2.0),
    )
    assert sol.result == optx.RESULTS.successful
    assert sol.value == 2


@pytest.mark.parametrize("slope", [-1, 1])
@pytest.mark.parametrize("root", [-1, 2])
def test_bisection_rejects_root_outside_interval(slope, root):
    with pytest.raises(eqx.EquinoxRuntimeError, match="root is not contained"):
        optx.root_find(
            lambda y, _: slope * (y - root),
            optx.Bisection(rtol=1e-5, atol=1e-5),
            0.5,
            options=dict(lower=0.0, upper=1.0),
        )
