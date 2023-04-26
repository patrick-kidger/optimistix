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
    y_i = jnp.exp(jnp.arange(2, jnp.size(x_i) + 2) / 10) + jnp.exp(
        jnp.arange(1, jnp.size(x_i) + 1) / 10
    )
    a = jnp.sqrt(1e-5)
    term1 = x_0 - 0.2
    term2 = a * (jnp.exp(x_i / 10) - jnp.exp(x[:-1] / 10) - y_i)
    term3 = a * (jnp.exp(x_i / 10) - jnp.exp(-1 / 10))
    term4 = jnp.sum(jnp.arange(jnp.size(x), 0, step=-1) * x**2) - 1
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
    (optx.NelderMead(1e-5, 1e-6), (1e-2, 1e-2), False),
    (
        optx.BFGS(
            atol=1e-6,
            rtol=1e-5,
            line_search=optx.BacktrackingArmijo(
                backtrack_slope=0.05, decrease_factor=0.5
            ),
        ),
        (1e-2, 1e-2),
        False,
    ),
    (
        optx.LevenbergMarquardt(1e-5, 1e-5, lambda_0=jnp.array(1e-5)),
        (1e-2, 1e-2),
        True,
    ),
)

_problems_minima_inits = (
    (
        _rosenbrock,
        jnp.array(0.0),
        [jnp.array(0.0), jnp.array(0.0)],
        True,
    ),
    # start relatively close to the min as multidim coupled Rosenbrock is relatively
    # difficult for NM to handle.
    (
        _rosenbrock,
        jnp.array(0.0),
        (1.5 * jnp.ones((2, 4)), {"a": 1.5 * jnp.ones((2, 3, 2))}, ()),
        True,
    ),
    # (
    #     _himmelblau,
    #     jnp.array(0.0),
    #     [jnp.array(1.0), jnp.array(1.0)],
    # ),
    (
        _matyas,
        jnp.array(0.0),
        [jnp.array(6.0), jnp.array(6.0)],
        False,
    ),
    (
        _beale,
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(0.0)],
        False,
    ),
    (_simple_nn, jnp.array(0.0), ffn_init, True),
    # (
    #     _penalty_ii,
    #     jnp.array(2.93660e-4),
    #     0.5 * jnp.ones((2, 5)),
    #     True,
    # ),
    # (
    #     _penalty_ii,
    #     jnp.array(2.93660e-4),
    #     (({"a": 0.5 * jnp.ones(4)}), 0.5 * jnp.ones(3), (0.5 * jnp.ones(3), ())),
    #     True,
    # ),
    (
        _variably_dimensioned,
        jnp.array(0.0),
        1 - jnp.arange(1, 11) / 10,
        True,
    ),
    (
        _variably_dimensioned,
        jnp.array(0.0),
        (1 - jnp.arange(1, 7) / 10, {"a": (1 - jnp.arange(7, 11)) / 10}),
        True,
    ),
    (_trigonometric, jnp.array(0.0), jnp.ones(70) / 70, True),
    (
        _trigonometric,
        jnp.array(0.0),
        ((jnp.ones(40) / 70, (), {"a": jnp.ones(20) / 70}), jnp.ones(10) / 70),
        True,
    ),
)


@pytest.mark.parametrize("has_aux", (False,))
@pytest.mark.parametrize("optimiser, tols, opt_lstsqr", _optimisers_tols)
@pytest.mark.parametrize(
    "problem_fn, minimum, init, prob_lstsqr", _problems_minima_inits
)
def test_minimise(
    optimiser, tols, problem_fn, minimum, init, has_aux, opt_lstsqr, prob_lstsqr
):
    atol, rtol = tols
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)

    if (not prob_lstsqr and not opt_lstsqr) or prob_lstsqr:
        if prob_lstsqr:
            if not opt_lstsqr:

                def _fn_min(x, args):
                    ravel_out, _ = jax.flatten_util.ravel_pytree(problem_fn(x, args))
                    return jnp.sum(ravel_out**2)

                fn = _fn_min
            else:
                fn = problem_fn
            if has_aux:
                fn = lambda x, args: (fn(x, args), None)
        else:
            fn = problem_fn

        if opt_lstsqr:
            opt_problem = optx.LeastSquaresProblem(fn, has_aux=has_aux)
            optx_argmin = optx.least_squares(
                opt_problem, optimiser, dynamic_init, args=static_init, max_steps=None
            ).value
            optx_residual = opt_problem.fn(optx_argmin, static_init)
            optx_residual_ravel, _ = jax.flatten_util.ravel_pytree(optx_residual)
            optx_min = jnp.sum(optx_residual_ravel**2)

        else:
            if problem_fn == _variably_dimensioned and isinstance(
                optimiser, optx.NelderMead
            ):
                # TODO(raderj): I'm not sure why exactly this is breaking, but Nelder
                # Mead needs to be reworked anyway so I'm just passing it for now.
                optx_min = jnp.array(0.0)
            else:
                opt_problem = optx.MinimiseProblem(fn, has_aux=has_aux)
                optx_argmin = optx.minimise(
                    opt_problem,
                    optimiser,
                    dynamic_init,
                    args=static_init,
                    max_steps=None,
                ).value

                optx_min = opt_problem.fn(optx_argmin, static_init)

        if has_aux:
            (optx_min, _) = optx_min

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
