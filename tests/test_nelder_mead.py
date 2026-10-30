import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest


def test_expansion_returns_new_best_vertex():
    # The initial simplex is [1, 1.1]. Reflection gives 0.9 and expansion
    # gives 0.8, so the new simplex is [1, 0.8].
    solver = optx.NelderMead(rtol=1e-8, atol=1e-8, rdelta=0.0, adelta=0.1)
    sol = optx.minimise(
        lambda x, _: x**2, solver, jnp.array(1.0), max_steps=2, throw=False
    )
    np.testing.assert_allclose(sol.value, 0.8, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.state.simplex, [1.0, 0.8], rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.state.best[0], sol.value**2)


def test_expansion_returns_updated_simplex():
    solver = optx.NelderMead(rtol=1e-8, atol=1e-8)
    fn = lambda x, _: (jnp.sum(x**2), None)
    y = jnp.array([1.0, 1.1])
    options = {"y0_simplex": True}
    tags = frozenset()
    state = solver.init(
        fn, y, None, options, jax.ShapeDtypeStruct((), y.dtype), None, tags
    )
    for _ in range(2):
        y, state, _ = solver.step(fn, y, None, options, state, tags)
    np.testing.assert_allclose(y, [1.0, 0.8], rtol=0, atol=1e-12)


def test_shrink_updates_worst_vertex():
    def fn(x, _):
        return x**2 + 16 * x**2 * (x - 1) ** 2

    # Start with [0, 1]. Reflection (-1) and contraction (0.5) are both
    # worse than the old worst point, so the simplex shrinks to [0, 0.5].
    solver = optx.NelderMead(rtol=1e-8, atol=1e-8, rdelta=0.0, adelta=1.0)
    sol = optx.minimise(fn, solver, jnp.array(0.0), max_steps=2, throw=False)
    assert sol.stats["num_shrinkages"] == 1
    np.testing.assert_allclose(sol.state.simplex, [0.0, 0.5])
    worst_f, worst_x, _ = sol.state.worst
    np.testing.assert_allclose(worst_x, 0.5)
    np.testing.assert_allclose(worst_f, fn(worst_x, None))


@pytest.mark.parametrize("kind", ["scalar", "vector", "pytree"])
def test_quadratic_minimum_and_gradient(kind):
    def solve(target):
        if kind == "scalar":
            y0 = jnp.array(1.0)
            fn = lambda y, a: (y - a) ** 2
        elif kind == "vector":
            y0 = jnp.array([1.0, 2.0])
            fn = lambda y, a: jnp.sum((y - a) ** 2)
        else:
            y0 = {"x": jnp.array(1.0), "y": jnp.array([2.0])}
            fn = lambda y, a: (y["x"] - a) ** 2 + jnp.sum((y["y"] - a) ** 2)
        solver = optx.NelderMead(rtol=1e-7, atol=1e-7)
        sol = optx.minimise(fn, solver, y0, args=target, max_steps=256)
        return sol

    target = jnp.array(0.25)
    sol = solve(target)
    assert sol.result == optx.RESULTS.successful
    for leaf in jax.tree.leaves(sol.value):
        np.testing.assert_allclose(leaf, target, atol=1e-5, rtol=0)

    def solution_sum(target):
        return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(solve(target).value))

    expected_gradient = 1.0 if kind == "scalar" else 2.0
    np.testing.assert_allclose(
        jax.grad(solution_sum)(target), expected_gradient, atol=1e-8
    )
