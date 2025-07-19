import jax
import jax.numpy as jnp
import optimistix._misc as optx_misc
import pytest

from .helpers import (
    correct_trees_to_combine,
    tree_allclose,
    trees_to_clip,
    wrong_trees_to_combine,
)


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


@pytest.mark.parametrize("pred, true, false, expected", correct_trees_to_combine)
def test_tree_where_correct_inputs(pred, true, false, expected):
    tree = optx_misc.tree_where(pred, true, false)
    assert tree_allclose(tree, expected)


@pytest.mark.parametrize("pred, true, false", wrong_trees_to_combine)
def test_tree_where_wrong_inputs(pred, true, false):
    with pytest.raises(AssertionError):
        optx_misc.tree_where(pred, true, false)
