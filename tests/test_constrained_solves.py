import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

from .constrained_problem_helpers import (
    bounded_paraboloids,
    constrained_quadratic_solvers,
    convex_constrained_minimisers,
    convex_constrained_paraboloids,
    least_squares_bounded_many_minima,
    minimise_bounded_with_local_minima,
    minimise_fn_y0_args_constraint_expected_result,
    nonconvex_constrained_minimisers,
)
from .helpers import gauss_newton_optimisers, tree_allclose


# TODO: I'm hard-coding the data type to float64 in many places, since my pytrees are
# often not arrays, and then allclose fails with a weak_type error. There is probably a
# better way to do this.


@pytest.mark.parametrize("fn, y0, args, bounds, expected_result", bounded_paraboloids)
def test_cauchy_newton(fn, y0, args, bounds, expected_result):
    sol = optx.quadratic_solve(
        fn,
        optx.CauchyNewton(rtol=1e-03, atol=1e-06),
        y0,
        args=args,
        bounds=bounds,
    )
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res)

    lower, upper = bounds
    check_bounds = jtu.tree_map(
        lambda x, l, u: jnp.logical_and(l <= x, x <= u), sol.value, lower, upper
    )
    assert jnp.all(jtu.tree_reduce(jnp.logical_and, check_bounds))


@pytest.mark.parametrize("solver", constrained_quadratic_solvers)
@pytest.mark.parametrize(
    "fn, y0, args, constraint, expected_result", convex_constrained_paraboloids
)
def test_interior_point(fn, y0, args, constraint, expected_result, solver):
    sol = optx.quadratic_solve(
        fn,
        solver,
        y0,
        args=args,
        constraint=constraint,
    )
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)  # TODO: improve this
    assert jnp.all(constraint(sol.value) >= 0)  # Solution is feasible


@pytest.mark.parametrize(
    "solver", convex_constrained_minimisers + nonconvex_constrained_minimisers
)
@pytest.mark.parametrize(
    "fn, y0, args, constraint, expected_result", convex_constrained_paraboloids
)
def test_constrained_minimisers(fn, y0, args, constraint, expected_result, solver):
    sol = optx.minimise(
        fn,
        solver,
        y0,
        args=args,
        constraint=constraint,
        max_steps=2**9,  # Needs more steps than unconstrained solvers
    )
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("solver", (optx.IPOPTLike(rtol=1e-3, atol=1e-3),))
@pytest.mark.parametrize(
    "fn, y0, args, constraint, expected_result",
    minimise_fn_y0_args_constraint_expected_result,
)
def test_ipoptlike(fn, y0, args, constraint, expected_result, solver):
    sol = optx.minimise(
        fn,
        solver,
        y0,
        args=args,
        constraint=constraint,
        max_steps=2**10,  # Needs more steps than unconstrained solvers
    )
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)


# TODO(jhaffner): SLSQP with Cauchy-Newton currently struggles with the harder problems.
# I think there is some subtlety in the inner solver/outer solver communication that
# needs ironing out. Tabled, but only for the short term!
bounded_minimisation_problems = (
    bounded_paraboloids  # minimise_bounded_with_local_minima
)


@pytest.mark.parametrize(
    "fn, y0, args, constraint, expected_result", bounded_minimisation_problems
)
def test_slsqp_cauchy_newton(fn, y0, args, constraint, expected_result):
    # TODO: create custom solver in helpers, and parametrize over it
    solver = optx.BFGS(rtol=1e-3, atol=1e-6, use_inverse=False)
    descent = optx.QuadraticSubproblemDescent(optx.CauchyNewton(rtol=1e-3, atol=1e-6))
    solver = eqx.tree_at(lambda s: s.descent, solver, descent)
    sol = optx.minimise(
        fn,
        solver,
        y0,
        args=args,
        bounds=constraint,
    )
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "solver", gauss_newton_optimisers + [optx.BFGS(rtol=1e-04, atol=1e-08)]
)
@pytest.mark.parametrize("fn, y0, args, bounds", least_squares_bounded_many_minima)
def test_gauss_newton_bfgs_clip(solver, fn, y0, args, bounds):
    # Check that the solution is within bounds for functions with minima outside bounds
    sol = optx.least_squares(
        fn,
        solver,
        y0,
        args=args,
        bounds=bounds,
        options={"clip": True},
    ).value
    lower, upper = bounds
    check_bounds = jtu.tree_map(
        lambda x, l, u: jnp.logical_and(l <= x, x <= u), sol, lower, upper
    )
    assert jnp.all(jtu.tree_reduce(jnp.logical_and, check_bounds))


# TODO: reverse the boundary map: we're only supporting clipping here
# TODO: do something about the solution.result?
@pytest.mark.parametrize(
    "fn, y0, args, bounds, expected_result", minimise_bounded_with_local_minima
)
def test_bfgs_clip(fn, y0, args, bounds, expected_result):
    boundary_map = optx.BoxProjection()
    sol = optx.minimise(
        fn,
        optx.BFGS(rtol=1e-03, atol=1e-06),
        y0,
        args=args,
        bounds=bounds,
        options={"boundary_map": boundary_map},
    )

    lower, upper = bounds
    check_bounds = jtu.tree_map(
        lambda x, l, u: jnp.logical_and(l <= x, x <= u), sol.value, lower, upper
    )
    assert jnp.all(jtu.tree_reduce(jnp.logical_and, check_bounds))

    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-3, atol=1e-6)


# TODO merge with other bounded solvers? This one is more capable and should work better
# on harder problems.
@pytest.mark.parametrize(
    "fn, y0, args, bounds, expected_result", minimise_bounded_with_local_minima
)
def test_coleman_li(fn, y0, args, bounds, expected_result):
    solver = optx.ColemanLi(rtol=1e-06, atol=1e-9)
    sol = optx.minimise(
        fn,
        solver,
        y0,
        args=args,
        bounds=bounds,
        max_steps=2**10,
    )

    lower, upper = bounds
    check_bounds = jtu.tree_map(
        lambda x, l, u: jnp.logical_and(l <= x, x <= u), sol.value, lower, upper
    )
    assert jnp.all(jtu.tree_reduce(jnp.logical_and, check_bounds))

    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-3, atol=1e-6)  # TODO tolerances?
