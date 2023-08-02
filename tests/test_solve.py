import jax

import optimistix as optx


def test_minimise():
    @jax.grad
    def f(offset):
        def fn(x, _):
            return x**2 + offset

        solver = optx.GradientDescent(learning_rate=0.1, rtol=0.1, atol=0.1)
        return optx.minimise(fn, solver, 0.0).value

    f(0.0)


def test_least_squares():
    @jax.grad
    def f(offset):
        def fn(x, _):
            return x + offset

        solver = optx.Dogleg(rtol=0.1, atol=0.1)
        return optx.least_squares(fn, solver, 0.0).value

    f(0.0)


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
