import jax.numpy as jnp
import optimistix as optx
import pytest


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


# https://github.com/patrick-kidger/optimistix/issues/33
@pytest.mark.parametrize(
    "solver", (optx.BFGS(rtol=1e-5, atol=1e-5), optx.NonlinearCG(atol=1e-5, rtol=1e-5))
)
def test_checks_last_point_minimiser(solver):
    def fn(y, _):
        return (y - 3.0) ** 2

    solver = optx.BestSoFarMinimiser(solver)
    sol = optx.minimise(fn, solver, jnp.array(0.0))
    assert sol.value == 3.0


def test_checks_last_point_least_squares():
    def fn(y, _):
        return y - 3.0

    solver = optx.BestSoFarLeastSquares(optx.GaussNewton(rtol=1e-5, atol=1e-5))
    sol = optx.least_squares(fn, solver, jnp.array(0.0))
    assert sol.value == 3.0
