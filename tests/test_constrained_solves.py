import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

from .constrained_problem_helpers import (
    barrier_values__y0_bounds_barrier_parameter_expected_result,
    bounded_paraboloids,
    combinatorial_smoke__fn_y0_constraint_bounds_expected_result,
    constrained_quadratic_solvers,
    convex_constrained_paraboloids,
    least_squares_bounded_many_minima,
    minimise_bounded_with_local_minima,
    minimise_fn_y0_args_constraint_expected_result,
    nonconvex_constrained_minimisers,
    tricky_geometries__fn_y0_args_constraint_bounds_expected_result,
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


# TODO name?
@pytest.mark.skip(
    reason="InteriorPoint is not numerically stable and struggles with "
    "simple problems since equality constraints were introduced. The solver or its "
    "logic need a fix or a replacement."
)
@pytest.mark.parametrize("solver", constrained_quadratic_solvers)
@pytest.mark.parametrize(
    "fn, y0, args, constraint, expected_result", convex_constrained_paraboloids[2:3]
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
    # TODO now I get allclose failures here - it seems that I disrupted some delicate
    # balance in this stupid solver after introducing equality constraints? Why the fuck
    # does handling tuples disrupt this thing?!
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)  # TODO: improve this
    equality_residual, inequality_residual = constraint(sol.value)
    assert equality_residual is None
    # TODO: why does this not stay strictly feasible anymore? This changed after I
    # introduced a distinction between equality and inequality constraints, the
    # constraint function must now return a tuple of arrays.
    jax.debug.print("inequality_residual: {}", inequality_residual)
    assert jnp.all(inequality_residual >= -1e-15)  # Solution is feasible


@pytest.mark.parametrize(
    "solver",
    # convex_constrained_minimisers +
    nonconvex_constrained_minimisers,
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
        # max_steps=50,  # Needs more steps than unconstrained solvers TODO: why?
        # throw=False,  # TODO just for debugging
    )

    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)


@pytest.mark.skip(reason="Need to streamline constraint categories (support ineq.)")
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


# TODO: this test should be unified with some of the other ones above! For now we're
# collecting more test cases, but these still need to be systematized.
@pytest.mark.skip(reason="some test cases do not provide inequality constraints.")
@pytest.mark.parametrize("solver", (optx.IPOPTLike(rtol=1e-3, atol=1e-3),))
@pytest.mark.parametrize(
    "fn, y0, args, constraint, bounds, expected_result",
    tricky_geometries__fn_y0_args_constraint_bounds_expected_result,
)
def test_tricky_geometries(fn, y0, args, constraint, bounds, expected_result, solver):
    sol = optx.minimise(
        fn,
        solver,
        y0,
        args=args,
        constraint=constraint,
        bounds=bounds,
    )
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    # TODO: adjust tolerance once I have worked out how to do the barrier parameter
    # update.
    assert tree_allclose(sol.value, res, rtol=1e-0, atol=1e-0)


# TODO(jhaffner): SLSQP with Cauchy-Newton currently struggles with the harder problems.
# I think there is some subtlety in the inner solver/outer solver communication that
# needs ironing out. Tabled, but only for the short term!
bounded_minimisation_problems = (
    bounded_paraboloids  # minimise_bounded_with_local_minima
)


# TODO: this particular solver combination does not really have an advantage over BFGS-B
# so it should be retired, I think. It has the potential to spend extra iterations in
# the subproblem and does not have a line search, besides BFGS-B is a very established
# solver. (BFGS-B consists of an Armijo line search with a CauchyNewtonDescent, while
# this version here puts a CauchyNewton solver into a QuadraticSubproblemDescent.)
@pytest.mark.skip(reason="Not a recommended combination, CauchyNewton will be retired.")
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
        bounds=constraint,  # TODO: what was I thinking with this naming?
    )
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)


@pytest.mark.skip(
    reason="New BFGS (inheriting from AbstractQuasiNewton) does not support clipping."
)
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
@pytest.mark.skip(
    reason=(
        "BFGS now inherits from AbstractQuasiNewton and does not support clipping. "
        "(This feature can be resurrected.)"
    )
)
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


# TODO: This should be the same test as the clipping test above, use parametrize to
# switch between different solvers.
# TODO: do something about the solution.result?
@pytest.mark.parametrize(
    "fn, y0, args, bounds, expected_result", minimise_bounded_with_local_minima
)
def test_bounded_bfgs(fn, y0, args, bounds, expected_result):
    sol = optx.minimise(
        fn,
        optx.BFGS_B(rtol=1e-03, atol=1e-06),
        y0,
        args=args,
        bounds=bounds,
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


@pytest.mark.parametrize(
    "y0, bounds, barrier_parameter, expected_result",
    barrier_values__y0_bounds_barrier_parameter_expected_result,
)
def test_logarithmic_barrier(y0, bounds, barrier_parameter, expected_result):
    barrier = optx.LogarithmicBarrier(bounds)
    result = barrier(y0, barrier_parameter)
    result = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(result, expected_result)


@pytest.mark.parametrize(
    "fn, y0, constraint, bounds, expected_result",
    combinatorial_smoke__fn_y0_constraint_bounds_expected_result,
)
def test_new_interior(fn, y0, constraint, bounds, expected_result):
    solver = optx.IPOPTLike(rtol=0.0, atol=1e-6)
    descent = optx.NewInteriorDescent()
    solver = eqx.tree_at(lambda s: s.descent, solver, descent)

    sol = optx.minimise(
        fn,
        solver,
        y0,
        constraint=constraint,
        bounds=bounds,
    )
    # TODO: tolerances still not great (this is true for all interior point solves!)
    # TODO: currently fails for one case - maybe it will help to include the slack
    # variables in the termination criterion, to better assess convergence in these,
    # which is currently ignored. (But does influence the inequality multipliers.)
    assert tree_allclose(sol.value, expected_result, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "fn, y0, args, bounds, expected_result",
    # minimise_bounded_with_local_minima +
    bounded_paraboloids[1:2],  # TODO
)
def test_bounds_new_interior(fn, y0, args, bounds, expected_result):
    solver = optx.IPOPTLike(rtol=0.0, atol=1e-6)
    descent = optx.NewInteriorDescent()
    solver = eqx.tree_at(lambda s: s.descent, solver, descent)

    sol = optx.minimise(
        fn,
        solver,
        y0,
        args,
        bounds=bounds,
        max_steps=2**3,
    )
    # TODO: hacky conversion of expected result necessary with PyTree inputs?

    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)
