import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from equinox.internal import ω

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
    return (100 * (x[1:] - x[:-1]), 1 - x[:-1])


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
    y_i = jnp.exp(jnp.arange(2, jnp.size(x) + 1) / 10) + jnp.exp(
        jnp.arange(1, jnp.size(x)) / 10
    )
    a = jnp.sqrt(1e-5)
    term1 = x_0 - 0.2
    term2 = a * (jnp.exp(x_i / 10) + jnp.exp(x[:-1] / 10) - y_i)
    term3 = a * (jnp.exp(x_i / 10) - jnp.exp(-1 / 10))
    term4 = jnp.sum(jnp.arange(jnp.size(x), 1, step=-1) * x[:-1] ** 2) - 1
    return (term1, term2, term3, term4)


def _variably_dimensioned(z, args):
    # TUOS eqn 25
    # least squares
    (x, _) = jax.flatten_util.ravel_pytree(z)
    increasing = jnp.arange(1, jnp.size(x) + 1)
    term1 = x - 1
    term2 = jnp.sum(increasing * (x - 1))
    term3 = term2**2
    return (term1, term2, term3)


def _trigonometric(z, args):
    # TUOS eqn 26
    # least squares
    (x, _) = jax.flatten_util.ravel_pytree(z)
    sumcos = jnp.sum(jnp.cos(x))
    increasing = jnp.arange(1, jnp.size(x) + 1, dtype=jnp.float32)
    return jnp.size(x) - sumcos + increasing * (1 - jnp.cos(x)) - jnp.sin(x)


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
        return (pred_y - y) ** 2

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

_optimisers_tols = (
    # (optx.NelderMead(1e-5, 1e-6), (1e-2, 1e-2)),
    # (
    #     optx.BFGS(
    #         atol=1e-6,
    #         rtol=1e-5,
    #         line_search=optx.BacktrackingArmijo(
    #             backtrack_slope=0.1, decrease_factor=0.5
    #         ),
    #     ),
    #     (1e-2, 1e-2),
    # ),
    (
        optx.NonlinearCG(
            1e-5,
            1e-5,
            optx.BacktrackingArmijo(backtrack_slope=0.05, decrease_factor=0.5),
            method=optx.dai_yuan,
        ),
        (1e-2, 1e-2),
    ),
)

_problems_minima_inits = (
    (
        optx.LeastSquaresProblem(_rosenbrock),
        jnp.array(0.0),
        [jnp.array(0.0), jnp.array(0.0)],
    ),
    # start relatively close to the min as multidim coupled Rosenbrock is
    # difficult for NM to handle.
    (
        optx.LeastSquaresProblem(_rosenbrock),
        jnp.array(0.0),
        (1.5 * jnp.ones((2, 4)), {"a": 1.5 * jnp.ones((2, 3, 2))}, ()),
    ),
    # the init seems to be finickey here? this is strange.
    (
        optx.MinimiseProblem(_himmelblau),
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(2.5)],
    ),
    (
        optx.MinimiseProblem(_matyas),
        jnp.array(0.0),
        [jnp.array(6.0), jnp.array(6.0)],
    ),
    (
        optx.MinimiseProblem(_beale),
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(0.0)],
    ),
    # (optx.LeastSquaresProblem(_simple_nn), jnp.array(0.0), ffn_init),
    # (
    #     optx.LeastSquaresProblem(_penalty_ii),
    #     jnp.array(2.93660e-4),
    #     0.5 * jnp.ones((2, 5)),
    # ),
    # (
    #     optx.LeastSquaresProblem(_penalty_ii),
    #     jnp.array(2.93660e-4),
    #     (({"a": 0.5 * jnp.ones(4)}), 0.5 * jnp.ones(3), (0.5 * jnp.ones(3), ())),
    # ),
    (
        optx.LeastSquaresProblem(_variably_dimensioned),
        jnp.array(0.0),
        1 - jnp.arange(1, 11) / 10,
    ),
    (
        optx.LeastSquaresProblem(_variably_dimensioned),
        jnp.array(0.0),
        (1 - jnp.arange(1, 7) / 10, {"a": (1 - jnp.arange(7, 11)) / 10}),
    ),
    (optx.LeastSquaresProblem(_trigonometric), jnp.array(0.0), jnp.ones(70) / 70),
    (
        optx.LeastSquaresProblem(_trigonometric),
        jnp.array(0.0),
        ((jnp.ones(40) / 70, (), {"a": jnp.ones(20) / 70}), jnp.ones(10) / 70),
    ),
)


import scipy


@pytest.mark.parametrize("problem, minimum, init", _problems_minima_inits)
def test_scipy(problem, minimum, init):
    if isinstance(problem, optx.LeastSquaresProblem):

        def scipy_problem(x):
            out_pytree = problem.fn(x, None)
            out_flat, _ = jax.flatten_util.ravel_pytree(out_pytree)
            return jnp.sum(out_flat**2)

    else:
        scipy_problem = lambda x: problem.fn(x, None)
    scipy_init, _ = jax.flatten_util.ravel_pytree(init)
    scipy_min = scipy.optimize.fmin_cg(scipy_problem, scipy_init)
    assert shaped_allclose(scipy_problem(scipy_min), minimum, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("has_aux", (False,))
@pytest.mark.parametrize("solver, tols", _optimisers_tols)
@pytest.mark.parametrize("problem, minimum, init", _problems_minima_inits)
def test_minimise(solver, tols, problem, minimum, init, has_aux):
    atol, rtol = tols
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)

    if isinstance(problem, optx.LeastSquaresProblem):
        optx_argmin = optx.least_squares(problem, solver, init, max_steps=None).value
        out = problem.fn(optx_argmin, static_init)
        out_ravel, _ = jax.flatten_util.ravel_pytree(out)
        optx_min = jnp.sum(out_ravel**2)
        assert shaped_allclose(optx_min, minimum, atol=atol, rtol=rtol)
    else:
        if not isinstance(solver, optx.AbstractLeastSquaresSolver):
            optx_argmin = optx.minimise(problem, solver, init, max_steps=None).value
            optx_min = problem.fn(optx_argmin, static_init)

            assert shaped_allclose(optx_min, minimum, atol=atol, rtol=rtol)


def _finite_difference_jvp(fn, primals, tangents):
    out = fn(*primals)
    # Choose ε to trade-off truncation error and floating-point rounding error.
    max_leaves = [jnp.max(jnp.abs(p)) for p in jtu.tree_leaves(primals)] + [1]
    scale = jnp.max(jnp.stack(max_leaves))
    ε = np.sqrt(np.finfo(np.float64).eps) * scale
    primals_ε = (ω(primals) + ε * ω(tangents)).ω
    out_ε = fn(*primals_ε)
    tangents_out = jtu.tree_map(lambda x, y: (x - y) / ε, out_ε, out)
    return out, tangents_out


# @pytest.mark.parametrize("has_aux", (False,))
# @pytest.mark.parametrize("optimiser, tols", _optimisers_tols)
# @pytest.mark.parametrize("problem_fn, minimum, init", _problems_minima_inits)
# def test_jvp(getkey, optimiser, tols, problem_fn, minimum, init, has_aux):
#     atol, rtol = tols
#     dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)

#     replace_random = lambda x: jr.normal(getkey(), x.shape)
#     t_dynamic_init = ω(dynamic_init).call(replace_random).ω

#     if has_aux:
#         fn = lambda x, args: (problem_fn(x, args), None)
#     else:
#         fn = problem_fn

#     opt_problem = optx.MinimiseProblem(fn, has_aux=has_aux)

#     def minimise(x):
#         return optx.minimise(
#             opt_problem, optimiser, x, args=static_init, max_steps=10_024
#         ).value

#     optx_argmin = minimise(dynamic_init)

#     expected_out, t_expected_out = _finite_difference_jvp(
#         minimise, (optx_argmin,), (t_dynamic_init,)
#     )
#     out, t_out = eqx.filter_jvp(minimise, (optx_argmin,), (t_dynamic_init,))

#     assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)
#     assert shaped_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)
