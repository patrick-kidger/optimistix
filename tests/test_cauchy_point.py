import jax.flatten_util as jfu
import jax.numpy as jnp
import pytest
from optimistix._solver.cauchy_point import cauchy_point

from .helpers import cauchy_point__y_bounds_grad_hessian_expected, tree_allclose


@pytest.mark.parametrize(
    "y, bounds, grad, hessian, expected",
    cauchy_point__y_bounds_grad_hessian_expected,
)
def test_cauchy_point(y, bounds, grad, hessian, expected):
    lower, upper = bounds
    cauchy = cauchy_point(y, lower, upper, grad, hessian)

    values, _ = jfu.ravel_pytree(cauchy)
    assert jnp.all(jnp.isfinite(values))

    lower_values, _ = jfu.ravel_pytree(lower)
    upper_values, _ = jfu.ravel_pytree(upper)

    assert jnp.all(jnp.where(values >= lower_values, True, False))
    assert jnp.all(jnp.where(values <= upper_values, True, False))

    assert tree_allclose(cauchy, expected)
