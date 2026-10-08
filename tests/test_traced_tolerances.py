import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from optimistix._misc import cauchy_termination, max_norm


@pytest.mark.parametrize("kind", ["root", "fixed_point", "minimise", "least_squares"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_traced_tolerances(kind, dtype):
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == jnp.float64)
    try:

        @jax.jit
        def solve(target, rtol, atol):
            y0 = jnp.array(0.0, dtype=dtype)
            if kind == "root":
                sol = optx.root_find(
                    lambda y, args: y - args,
                    optx.Newton(rtol=rtol, atol=atol),
                    y0,
                    args=target,
                )
            elif kind == "fixed_point":
                sol = optx.fixed_point(
                    lambda y, args: (y + args) / 2,
                    optx.FixedPointIteration(rtol=rtol, atol=atol),
                    y0,
                    args=target,
                )
            elif kind == "minimise":
                sol = optx.minimise(
                    lambda y, args: (y - args) ** 2,
                    optx.BFGS(rtol=rtol, atol=atol),
                    y0,
                    args=target,
                )
            else:
                sol = optx.least_squares(
                    lambda y, args: y - args,
                    optx.GaussNewton(rtol=rtol, atol=atol),
                    y0,
                    args=target,
                )
            return sol.value

        for target in (1.0, -2.0):
            actual = solve(
                jnp.array(target, dtype=dtype),
                jnp.array(1e-5, dtype=dtype),
                jnp.array(1e-5, dtype=dtype),
            )
            assert jnp.allclose(actual, target, rtol=1e-4, atol=1e-4)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


@pytest.mark.parametrize(
    "rtol, atol, expected", [(0.01, 0.001, False), (0.1, 0.001, True), (0.0, 0.2, True)]
)
def test_traced_cauchy_tolerances(rtol, atol, expected):
    @jax.jit
    def terminate(rtol, atol):
        return cauchy_termination(
            rtol,
            atol,
            max_norm,
            jnp.array(2.0),
            jnp.array(0.1),
            jnp.array(3.0),
            jnp.array(0.1),
        )

    assert terminate(jnp.array(rtol), jnp.array(atol)) == expected
