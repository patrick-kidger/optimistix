import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix._misc as optx_misc
import pytest

from .constrained_problem_helpers import trees_to_clip
from .helpers import tree_allclose


def test_inexact_asarray_no_copy():
    x = jnp.array([1.0])
    assert optx_misc.inexact_asarray(x) is x
    y = jnp.array([1.0, 2.0])
    assert jax.vmap(optx_misc.inexact_asarray)(y) is y


# See JAX issue #15676
def test_inexact_asarray_jvp():
    p, t = jax.jvp(optx_misc.inexact_asarray, (1.0,), (2.0,))
    assert type(p) is not float
    assert type(t) is not float


@pytest.mark.parametrize("tree, lower, upper, result", trees_to_clip)
def test_tree_clip(tree, lower, upper, result):
    clipped_tree = optx_misc.tree_clip(tree, lower, upper)
    assert tree_allclose(clipped_tree, result)


@pytest.mark.parametrize("y, lower, upper, _", trees_to_clip)
def test_checked_bounds(y, lower, upper, _):
    # checked_bounds should raise an error if lower > upper for variable bounds...
    with pytest.raises(eqx.EquinoxRuntimeError):
        optx_misc.checked_bounds(y, (upper, lower))
    # ...it should also raise an error if y is not within the bounds.
    with pytest.raises(eqx.EquinoxRuntimeError):
        optx_misc.checked_bounds(upper, (lower, y))
    with pytest.raises(eqx.EquinoxRuntimeError):
        optx_misc.checked_bounds(lower, (y, upper))
    # Finally, it should raise an error if the tree structures do not match.
    with pytest.raises(ValueError):
        optx_misc.checked_bounds(y, (lower, ()))
