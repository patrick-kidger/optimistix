import jax.numpy as jnp

import optimistix as optx


def test_fixed_point():
    def fn(y, _):
        return jnp.tanh(y + 1)

    solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
    solver = optx.BestSoFarFixedPoint(solver)
    sol = optx.fixed_point(fn, solver, jnp.array(0.0))
    assert jnp.allclose(sol.value, 0.96118069, rtol=1e-5, atol=1e-5)


def test_root_find():
    def fn(y, _):
        return y - jnp.tanh(y + 1)

    solver = optx.Bisection(rtol=1e-6, atol=1e-6)
    solver = optx.BestSoFarRootFinder(solver)
    options = dict(lower=-1, upper=1)
    sol = optx.root_find(fn, solver, jnp.array(0.0), options=options)
    assert jnp.allclose(sol.value, 0.96118069, rtol=1e-5, atol=1e-5)


def test_least_squares():
    def fn(y, _):
        return y - jnp.tanh(y + 1)

    solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
    solver = optx.BestSoFarLeastSquares(solver)
    sol = optx.least_squares(fn, solver, jnp.array(0.0))
    assert jnp.allclose(sol.value, 0.96118069, rtol=1e-5, atol=1e-5)


def test_minimise():
    def fn(y, _):
        return 0.5 * (y - jnp.tanh(y + 1)) ** 2

    solver = optx.BFGS(rtol=1e-6, atol=1e-6)
    solver = optx.BestSoFarMinimiser(solver)
    sol = optx.minimise(fn, solver, jnp.array(0.0))
    assert jnp.allclose(sol.value, 0.96118069, rtol=1e-5, atol=1e-5)
