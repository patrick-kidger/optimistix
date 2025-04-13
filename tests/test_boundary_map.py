import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest

from .constrained_problem_helpers import (
    bounded_paraboloids,
    minimise_bounded_with_local_minima,
    outside_nonconvex_set__y_constraint_bounds_expected_result,
    trees_to_clip,
)
from .helpers import tree_allclose


@pytest.mark.parametrize("tree, lower, upper, result", trees_to_clip)
def test_box_projection(tree, lower, upper, result):
    boundary_map = optx.BoxProjection()
    constraint = None
    projected_tree, _ = boundary_map(tree, constraint, (lower, upper))
    assert tree_allclose(projected_tree, result)


@pytest.mark.parametrize(
    "fn, y0, args, bounds, expected_result",
    bounded_paraboloids + minimise_bounded_with_local_minima,
)
def test_projected_gradient_box_projection(fn, y0, args, bounds, expected_result):
    if fn.__name__ == "scalar_rosenbrock":
        pytest.skip("The solver begins to cycle around the minimum (+/- 1e-1 on x).")
    boundary_map = optx.BoxProjection()
    gradient_descent = optx.GradientDescent(learning_rate=0.002, rtol=1e-9, atol=1e-12)
    solver = optx.ProjectedGradientDescent(
        solver=gradient_descent, boundary_map=boundary_map
    )
    # Gradient descent requires many steps on the challenging benchmark problems
    sol = optx.minimise(fn, solver, y0, args, bounds=bounds, max_steps=100_000)
    res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert tree_allclose(sol.value, res, rtol=1e-2, atol=1e-2)


@pytest.mark.skip(
    reason="Closest feasible point now expects clip=True, but not all test cases "
    "provide bounds. This was implemented as a way to experiment with IPOPTLike, and "
    "needs to be changed once the dust settles on the IPOPTLike API, and projections "
    "more generally."
)
@pytest.mark.parametrize(
    "y, constraint, bounds, expected_result",
    outside_nonconvex_set__y_constraint_bounds_expected_result,
)
def test_closest_feasible_point(y, constraint, bounds, expected_result):
    solver = optx.BFGS(rtol=1e-6, atol=1e-12)  # This is not an ideal solver here
    boundary_map = optx.ClosestFeasiblePoint(1e-2, solver)
    restored_point, result = boundary_map(y, constraint, bounds)
    # res = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), expected_result)
    assert result == optx.RESULTS.successful
    # TODO check if the restored point is close to the expected result
    equality_residual, inequality_residual = constraint(restored_point)
    assert jnp.all(inequality_residual >= -1e-6)  # TODO tolerance?
    # assert tree_allclose(restored_point, res, rtol=1e-2, atol=1e-2)
