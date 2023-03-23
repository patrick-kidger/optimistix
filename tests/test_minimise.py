from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from equinox.internal import ω
from jaxtyping import ArrayLike

import optimistix as optx

from .helpers import shaped_allclose


def _square(x):
    return ω(x).call(lambda x: x**2).ω


#
# Some standard test functions for nonlinear minimisation. All of these can be found
# on the wikipedia page: https://en.wikipedia.org/wiki/Test_functions_for_optimization
#


def _himmelblau(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = (x**2 + y - 11.0) ** 2
    term2 = (x + y**2 - 7.0) ** 2
    return term1 + term2


def _matyas(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = 0.26 * (x**2 + y**2)
    term2 = 0.48 * x * y
    return term1 - term2


def _eggholder(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = -(y + 47.0) * jnp.sin(jnp.sqrt(jnp.abs(0.5 * x + (y + 47.0))))
    term2 = -x * jnp.sin(jnp.sqrt(jnp.abs(x - (y + 47.0))))
    return term1 + term2


def _beale(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y**2) ** 2
    term3 = (2.625 - x + x * y**3) ** 2
    return term1 + term2 + term3


def _penalty_i(z, args):
    # eqn 23 of Testing Unconstrained Minimization Software
    pass


def _penalty_ii(z, args):
    # eqn 23 of Testing Unconstrained Minimization Software
    pass


def _variably_dimesnioned(z, args):
    # eqn 25 of Testing Unconstrained Minimization Software
    pass


def _trigonometric(z, args):
    # eqn 26
    pass


def _indicator_nn(model, args):
    key = jr.PRNGKey(17)
    model_key, data_key = jr.split(key, 3)
    x = jr.normal(data_key, (100, 2))
    y = jnp.where(x > 0, 1, 0)
    breakpoint()

    def loss(model, x, y):
        pred_y = eqx.filter_vmap(x)
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

    return loss(model, x, y)


class FFN(eqx.Module):
    layers: List[eqx.nn.Linear]
    out_bias: ArrayLike

    def __init__(self, *, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = [eqx.nn.Linear(2, 8, key=key1), eqx.nn.Linear(8, 1, key=key2)]
        self.out_bias = jnp.ones(1)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jax.nn.sigmoid(self.layers[-1](x) + self.out_bias)


#
# Some nonlinear least squares test problems. These are taken from
# J. Moré, B. Garbow, and K. Hillstrom "Testing Unconstrained Optimization Software:"
# https://www.osti.gov/servlets/purl/6650344.
# See also the Fortran package TEST_NLS:
# https://people.math.sc.edu/Burkardt/f_src/test_nls/test_nls.html.
#

fixed_key = jr.PRNGKey(42)

_fn_and_init = (
    (
        optx.MinimiseProblem(_himmelblau, False),
        [jnp.array(1.0), jnp.array(1.0)],
        [jnp.array(3.0), jnp.array(2.0)],
    ),
    (
        optx.MinimiseProblem(_matyas, False),
        [6.0 * jnp.array(1.0), 6.0 * jnp.array(1.0)],
        ([jnp.array(0.0), jnp.array(0.0)]),
    ),
    (
        optx.MinimiseProblem(_beale, False),
        [jnp.array(2.0), jnp.array(0.0)],
        [jnp.array(3.0), jnp.array(0.5)],
    ),
    # (
    #     _indicator_nn,
    #     FFN(key=fixed_key),
    #     FFN(key=fixed_key)
    # )
)


#
# TODO(Jason): add tests for least_square solvers as well.
#

_minimisers = ((optx.NelderMead(1e-4, 1e-5)),)


@pytest.mark.parametrize("solver", _minimisers)
@pytest.mark.parametrize("problem, init, result_exact", _fn_and_init)
def test_minimise(solver, problem, init, result_exact):
    solver.init(problem, init, None, {})
    result_optx = optx.minimise(problem, solver, init, max_steps=1024)

    assert shaped_allclose(result_optx.value, result_exact, atol=1e-4, rtol=1e-5)
