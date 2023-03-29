import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import optimistix as optx

from .helpers import shaped_allclose


#
# Some standard test functions for nonlinear minimisation. Many of these can be found
# on the wikipedia page: https://en.wikipedia.org/wiki/Test_functions_for_optimization
# (Wiki.) The rest can be found in Testing Unconstrained Optimization Software (TUOS)
# by More, Garbow, and Hillstrom.
# See also the Fortran package TEST_NLS:
# https://people.math.sc.edu/Burkardt/f_src/test_nls/test_nls.html.
#


def _rosenbrock(z, args):
    # Wiki
    # least squares
    (x, _) = jax.flatten_util.ravel_pytree(z)
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def _himmelblau(z, args):
    # Wiki
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = (x**2 + y - 11.0) ** 2
    term2 = (x + y**2 - 7.0) ** 2
    return term1 + term2


def _matyas(z, args):
    # Wiki
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = 0.26 * (x**2 + y**2)
    term2 = 0.48 * x * y
    return term1 - term2


def _eggholder(z, args):
    # Wiki
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = -(y + 47.0) * jnp.sin(jnp.sqrt(jnp.abs(0.5 * x + (y + 47.0))))
    term2 = -x * jnp.sin(jnp.sqrt(jnp.abs(x - (y + 47.0))))
    return term1 + term2


def _beale(z, args):
    # Wiki
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y**2) ** 2
    term3 = (2.625 - x + x * y**3) ** 2
    return term1 + term2 + term3


def _penalty_ii(z, args):
    # TUOS eqn 24
    # least squares
    (x, _) = jax.flatten_util.ravel_pytree(z)
    x_0, x_i = x[0], x[1:]
    y_i = jnp.exp(jnp.arange(2, jnp.size(x_i) + 2) / 10) + jnp.exp(
        jnp.arange(1, jnp.size(x_i) + 1) / 10
    )
    a = jnp.sqrt(1e-5)
    term1 = x_0 - 0.2
    term2 = a * (jnp.exp(x_i / 10) - jnp.exp(x[:-1] / 10) - y_i)
    term3 = a * (jnp.exp(x_i / 10) - jnp.exp(-1 / 10))
    term4 = jnp.sum(jnp.arange(jnp.size(x), 0, step=-1) * x**2) - 1
    return term1**2 + jnp.sum(term2**2) + jnp.sum(term3**2) + term4**2


def _variably_dimensioned(z, args):
    # TUOS eqn 25
    # least squares
    (x, _) = jax.flatten_util.ravel_pytree(z)
    increasing = jnp.arange(1, jnp.size(x) + 1)
    term1 = jnp.sum(x - 1)
    term2 = jnp.sum(increasing * (x - 1))
    term3 = term2**2
    return term1**2 + term2**2 + term3**2


def _trigonometric(z, args):
    # TUOS eqn 26
    # least squares
    (x, _) = jax.flatten_util.ravel_pytree(z)
    sumcos = jnp.sum(jnp.cos(x))
    increasing = jnp.arange(1, jnp.size(x) + 1, dtype=jnp.float32)
    term1 = jnp.size(x) - sumcos + increasing * (1 - jnp.cos(x)) - jnp.sin(x)
    return jnp.sum(term1**2)


def _simple_nn(model, args):
    # args is the activation functions in
    # the MLP.
    model = eqx.combine(model, args)

    key = jr.PRNGKey(17)
    model_key, data_key = jr.split(key, 2)
    x = jnp.linspace(0, 1, 100)[..., None]
    y = x**2

    def loss(model, x, y):
        pred_y = eqx.filter_vmap(model)(x)
        return jnp.sum((pred_y - y) ** 2)

    return loss(model, x, y)


ffn_init = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=1, key=jr.PRNGKey(17))

layer1, layer2 = ffn_init.layers

weight1 = jnp.array(
    [
        [3.39958394],
        [-0.00648151],
        [-0.57178741],
        [-0.9259611],
        [0.43010087],
        [0.74950293],
        [-0.35932756],
        [1.28393738],
    ]
)
bias1 = jnp.array(
    [
        -0.14702571,
        3.61400359,
        -1.01404942,
        0.3223137,
        2.19991259,
        -0.51860872,
        0.4717813,
        0.9449359,
    ]
)
weight2 = jnp.array(
    [
        [
            0.26476087,
            0.05913492,
            0.40830152,
            0.704758,
            0.64347694,
            0.85807767,
            0.54285774,
            0.05507022,
        ]
    ]
)
bias2 = jnp.array([-2.16578771])

#
# The MLP can be difficult for some of the solvers to optimise. Rather than set
# max_steps to a higher value and iterate for longer, we initialise the MLP
# closer to the minimum by explicitly setting the weights and biases.
#


def get_weights(model):
    layer1, layer2 = model.layers
    return layer1.weight, layer1.bias, layer2.weight, layer2.bias


ffn_init = eqx.tree_at(get_weights, ffn_init, (weight1, bias1, weight2, bias2))

_optimisers_tols = ((optx.NelderMead(1e-5, 1e-6), (1e-2, 1e-2)),)

_problems_minima_inits = (
    (
        optx.MinimiseProblem(_rosenbrock, False),
        jnp.array(0.0),
        [jnp.array(0.0), jnp.array(0.0)],
    ),
    # start relatively close to the min as multidim coupled Rosenbrock is relatively
    # difficult for NM to handle.
    (
        optx.MinimiseProblem(_rosenbrock, False),
        jnp.array(0.0),
        (1.5 * jnp.ones((2, 4)), {"a": 1.5 * jnp.ones((2, 3, 2))}, ()),
    ),
    (
        optx.MinimiseProblem(_himmelblau, False),
        jnp.array(0.0),
        [jnp.array(1.0), jnp.array(1.0)],
    ),
    (
        optx.MinimiseProblem(_matyas, False),
        jnp.array(0.0),
        [jnp.array(6.0), jnp.array(6.0)],
    ),
    (
        optx.MinimiseProblem(_beale, False),
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(0.0)],
    ),
    (optx.MinimiseProblem(_simple_nn, False), jnp.array(0.0), ffn_init),
    (
        optx.MinimiseProblem(_penalty_ii, False),
        jnp.array(2.93660e-4),
        0.5 * jnp.ones((2, 5)),
    ),
    (
        optx.MinimiseProblem(_penalty_ii, False),
        jnp.array(2.93660e-4),
        (({"a": 0.5 * jnp.ones(4)}), 0.5 * jnp.ones(3), (0.5 * jnp.ones(3), ())),
    ),
    (
        optx.MinimiseProblem(_variably_dimensioned, False),
        jnp.array(0.0),
        1 - jnp.arange(1, 11) / 10,
    ),
    (
        optx.MinimiseProblem(_variably_dimensioned, False),
        jnp.array(0.0),
        (1 - jnp.arange(1, 7) / 10, {"a": (1 - jnp.arange(7, 11)) / 10}),
    ),
    (optx.MinimiseProblem(_trigonometric, False), jnp.array(0.0), jnp.ones(70) / 70),
    (
        optx.MinimiseProblem(_trigonometric, False),
        jnp.array(0.0),
        ((jnp.ones(40) / 70, (), {"a": jnp.ones(20) / 70}), jnp.ones(10) / 70),
    ),
)


#
# TODO(raderj): add tests for least_square solvers as well.
#


@pytest.mark.parametrize("optimiser, tols", _optimisers_tols)
@pytest.mark.parametrize("problem, minimum, init", _problems_minima_inits)
def test_minimise(optimiser, tols, problem, minimum, init):
    atol, rtol = tols
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    result_optx = optx.minimise(
        problem, optimiser, dynamic_init, args=static_init, max_steps=10_024
    )
    minimum_optx = problem.fn(result_optx.value, static_init)

    assert shaped_allclose(minimum_optx, minimum, atol=atol, rtol=rtol)
