from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

import optimistix as optx


#
# MINIMISATION PROBLEMS
#
# Some standard test functions for nonlinear minimisation. Many of these can be found
# on the wikipedia page: https://en.wikipedia.org/wiki/Test_functions_for_optimization
# (Wiki.) The rest can be found in Testing Unconstrained Optimization Software (TUOS)
# by Moré, Garbow, and Hillstrom.
# See also the Fortran package TEST_NLS:
# https://people.math.sc.edu/Burkardt/f_src/test_nls/test_nls.html.
#


def _bowl(tree: PyTree[Array], args: Any):
    # Trivial quadratic bowl smoke test for convergence
    (y, _) = jax.flatten_util.ravel_pytree(tree)
    size = y.size
    key = jr.PRNGKey(17)
    matrix = jr.normal(key, (size, size))
    pd_matrix = matrix.T @ matrix
    w, v = jnp.linalg.eig(pd_matrix)
    return y.T @ pd_matrix @ y


def _uncoupled_simple(tree: PyTree[Array], args: Any):
    leaves, treedef = jtu.tree_flatten(tree)
    key = jr.PRNGKey(17)
    random_positive = treedef.unflatten(
        [jr.normal(key, leaf.shape, leaf.dtype) ** 2 for leaf in leaves]
    )
    return (ω(tree).call(jnp.square) * random_positive**ω).ω


def _rosenbrock(tree: PyTree[Array], args: Any):
    # Wiki
    # least squares
    (y, _) = jax.flatten_util.ravel_pytree(tree)
    return (100 * (y[1:] - y[:-1]), 1 - y[:-1])


def _himmelblau(tree: PyTree[Array], args: Any):
    # Wiki
    (y, z) = tree
    term1 = ((ω(y).call(jnp.square) + z**ω - 11.0) ** 2).ω
    term2 = ((y**ω + ω(z).call(jnp.square) - 7.0) ** 2).ω
    return (term1**ω + term2**ω).ω


def _matyas(tree: PyTree[Array], args: Any):
    # Wiki
    (y, z) = tree
    term1 = (0.26 * (ω(y).call(jnp.square) + ω(z).call(jnp.square))).ω
    term2 = (0.48 * y**ω * z**ω).ω
    return (term1**ω - term2**ω).ω


def _eggholder(tree: PyTree[Array], args: Any):
    # Wiki
    (y, z) = tree
    # the composition sin . sqrt . abs
    ssa = lambda x: jnp.sin(jnp.sqrt(jnp.abs(x)))
    term1 = (-(z**ω + 47.0) * (0.5 * y**ω + z**ω + 47.0).call(ssa)).ω
    term2 = (-(y**ω) * (y**ω - (z**ω - 47.0)).call(ssa)).ω
    return (term1**ω + term2**ω).ω


def _beale(tree: PyTree[Array], args: Any):
    # Wiki
    (y, z) = tree
    term1 = ((1.5 - y**ω + y**ω * z**ω) ** 2).ω
    term2 = ((2.25 - y**ω + y**ω * ω(z).call(jnp.square)) ** 2).ω
    term3 = ((2.625 - y**ω + y**ω * ω(z).call(lambda x: x**3)) ** 2).ω
    return (term1**ω + term2**ω + term3**ω).ω


def _penalty_ii(tree: PyTree[Array], args: Any):
    # TUOS eqn 24
    # least squares
    (y, _) = jax.flatten_util.ravel_pytree(tree)
    y_0, y_i = y[0], y[1:]
    y_i = jnp.exp(jnp.arange(2, jnp.size(y) + 1) / 10) + jnp.exp(
        jnp.arange(1, jnp.size(y)) / 10
    )
    a = jnp.sqrt(1e-5)
    term1 = y_0 - 0.2
    term2 = a * (jnp.exp(y_i / 10) + jnp.exp(y[:-1] / 10) - y_i)
    term3 = a * (jnp.exp(y_i / 10) - jnp.exp(-1 / 10))
    term4 = jnp.sum(jnp.arange(jnp.size(y), 1, step=-1) * y[:-1] ** 2) - 1
    return (term1, term2, term3, term4)


def _variably_dimensioned(tree: PyTree[Array], args: Any):
    # TUOS eqn 25
    # least squares
    (y, _) = jax.flatten_util.ravel_pytree(tree)
    increasing = jnp.arange(1, jnp.size(y) + 1)
    term1 = y - 1
    term2 = jnp.sum(increasing * (y - 1))
    term3 = term2**2
    return (term1, term2, term3)


def _trigonometric(tree: PyTree[Array], args: Any):
    # TUOS eqn 26
    # least squares
    (y, _) = jax.flatten_util.ravel_pytree(tree)
    sumcos = jnp.sum(jnp.cos(y))
    increasing = jnp.arange(1, jnp.size(y) + 1, dtype=jnp.float32)
    return jnp.size(y) - sumcos + increasing * (1 - jnp.cos(y)) - jnp.sin(y)


def _simple_nn(model: PyTree[Array], args: Any):
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

# TODO(raderj): patch or remove _penalty_ii
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

least_squares_problem_minima_init = (
    (
        optx.LeastSquaresProblem(_uncoupled_simple),
        jnp.array(0.0),
        ({"a": 0.05 * jnp.ones((2, 3, 3))}, (0.05 * jnp.ones(2))),
    ),
    # (
    #     optx.LeastSquaresProblem(_rosenbrock),
    #     jnp.array(0.0),
    #     [jnp.array(0.0), jnp.array(0.0)],
    # ),
    # (
    #     optx.LeastSquaresProblem(_rosenbrock),
    #     jnp.array(0.0),
    #     (1.5 * jnp.ones((2, 4)), {"a": 1.5 * jnp.ones((2, 3, 2))}, ()),
    # ),
    # (optx.LeastSquaresProblem(_simple_nn), jnp.array(0.0), ffn_init),
    # (
    #     optx.LeastSquaresProblem(_variably_dimensioned),
    #     jnp.array(0.0),
    #     1 - jnp.arange(1, 11) / 10,
    # ),
    # (
    #     optx.LeastSquaresProblem(_variably_dimensioned),
    #     jnp.array(0.0),
    #     (1 - jnp.arange(1, 7) / 10, {"a": (1 - jnp.arange(7, 11) / 10)}),
    # ),
    # (optx.LeastSquaresProblem(_trigonometric), jnp.array(0.0), jnp.ones(70) / 70),
    # (
    #     optx.LeastSquaresProblem(_trigonometric),
    #     jnp.array(0.0),
    #     ((jnp.ones(40) / 70, (), {"a": jnp.ones(20) / 70}), jnp.ones(10) / 70),
    # ),
)

minimisation_problem_minima_init = (
    (
        optx.MinimiseProblem(_bowl),
        jnp.array(0.0),
        ({"a": 0.05 * jnp.ones((2, 3, 3))}, (0.05 * jnp.ones(2))),
    ),
    # (
    #     optx.MinimiseProblem(_himmelblau),
    #     jnp.array(0.0),
    #     [jnp.array(2.0), jnp.array(2.5)],
    # ),
    # (
    #     optx.MinimiseProblem(_matyas),
    #     jnp.array(0.0),
    #     [jnp.array(6.0), jnp.array(6.0)],
    # ),
    # (
    #     optx.MinimiseProblem(_beale),
    #     jnp.array(0.0),
    #     [jnp.array(2.0), jnp.array(0.0)],
    # ),
)

# ROOT FIND/FIXED POINT PROBLEMS
#
# These are mostly derived from common problems arising in
# ordinary and partial differential equations. See
# Diffrax for more on differential equations:
# https://github.com/patrick-kidger/diffrax.
#
# - Trivial smoke tests
# - MLP(y) - 10 y
# - Nonlinear heat equation with Crank Nicholson
# - Implicit midpoint Runge-Kutta in y-space, f-space, and k-space
#


# SETUP
# Helper functions, most of which are single steps of a differential
# equation solve.
def _getsize(y: PyTree[Array]):
    return jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(jnp.size, y))


def _laplacian(y: PyTree[Array], dx: Scalar):
    (y, unflatten) = jax.flatten_util.ravel_pytree(y)
    laplacian = jnp.zeros_like(y)
    laplacian = laplacian.at[1:-1].set((y[2:] + y[1:-1] + y[:-2]) / dx)
    return unflatten(y)


def _nonlinear_heat_pde_general(
    y0: PyTree[Array],
    f0: PyTree[Array],
    dx: Scalar,
    t0: Scalar,
    t1: Scalar,
    y: PyTree[Array],
    args: Any,
):
    # A single step time `t0` to time `t1` of the Crank Nicolson scheme applied to
    # the nonlinear heat equation:
    # `d y(t, x)/dt = (1 - y(t, x)) Δy(t, x)`
    # where `d/dt` is a partial derivative wrt time and `Δ` is the Laplacian.
    stepsize = t1 - t0
    f_val = ((1 - y**ω) * _laplacian(y, dx) ** ω).ω
    return (y0**ω + 0.5 * stepsize * (f_val**ω + f0**ω)).ω


# Note that the midpoint methods below assume that `f` is autonomous.


def _midpoint_y_general(
    f: Callable[[PyTree[Array], Any], PyTree[Array]],
    y0: PyTree[Array],
    dt: Scalar,
    y: PyTree[Array],
    args: Any,
):
    # Solve an implicit midpoint Runge-Kutta step with fixed point iteration
    # in "y space," ie. the typical representation of a Runge-Kutta method
    f_new = f((0.5 * (y0**ω + y**ω)).ω, args)
    return (y0**ω + dt * f_new**ω).ω


def _midpoint_f_general(
    f: Callable[[PyTree[Array], Any], PyTree[Array]],
    y0: PyTree[Array],
    dt: Scalar,
    y: PyTree[Array],
    args: Any,
):
    # Solve an implicit Runge-Kutta step with fixed point iteration
    # in "f-space," ie. we apply `f` to both sides of the Runge-Kutta method
    # and do the fixed-point iteration in this space.
    return f((y0**ω + dt * y**ω).ω, args)


def _midpoint_k_general(
    f: Callable[[PyTree[Array], Any], PyTree[Array]],
    y0: PyTree[Array],
    dt: Scalar,
    y: PyTree[Array],
    args: Any,
):
    # Solve an implicit midpoint Runge-Kutta step with fixed point iteration
    # in "k-space," ie. we do the fixed point iterations directly on the `k` terms
    # (see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
    # of the Runge-Kutta method.
    # Note that in practice this just means we are finding `f * dt`
    # instead of `f`.
    return (f((y0**ω + y**ω).ω, args) ** ω * dt).ω


class Robertson(eqx.Module):
    # Take directly from the Diffrax docs, but modified to accept and return
    # a PyTree and have no `t` argument.
    k1: float
    k2: float
    k3: float

    def __call__(self, y, args):
        (y, unflatten) = jax.flatten_util.ravel_pytree(y)
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return unflatten(jnp.stack([f0, f1, f2]))


# PROBLEMS
def _exponential(tree: PyTree[Array], args: Any):
    return jtu.tree_map(lambda x: jnp.exp(-x), tree)


def _sin(tree: PyTree[Array], args: Any):
    return jtu.tree_map(jnp.sin, tree)


def _nn(tree: PyTree[Array], args: Any):
    (y, unflatten) = jax.flatten_util.ravel_pytree(tree)
    size = _getsize(y)
    model = eqx.nn.MLP(
        in_size=size,
        out_size=size,
        width_size=8,
        depth=1,
        activation=jax.nn.softplus,  # more complex than relu
        key=jr.PRNGKey(42),
    )
    key = jr.PRNGKey(17)
    model_key, data_key = jr.split(key, 2)
    return unflatten(0.1 * model(y))


def _nonlinear_heat_pde(y: PyTree[Array], args: Any):
    # solve the nonlinear heat equation as above from "0" to "1" in a
    # single step.
    x = jnp.linspace(-1, 1, 100)
    dx = x[1] - x[0]
    y0 = x**2
    f0 = ((1 - y0**ω) * _laplacian(y0, dx) ** ω).ω
    return _nonlinear_heat_pde_general(
        y0, f0, dx, jnp.array(0.0), jnp.array(1.0), y, args
    )


def _midpoint_y_linear(tree: PyTree[Array], args: Any):
    (y, unflatten) = jax.flatten_util.ravel_pytree(tree)
    size = y.size
    key = jr.PRNGKey(17)
    matrix = jr.normal(key, (size, size))
    f = lambda x, _: matrix @ x
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1 / (2**4))
    return unflatten(_midpoint_y_general(f, y0, dt, y, args))


def _midpoint_f_linear(tree: PyTree[Array], args: Any):
    (y, unflatten) = jax.flatten_util.ravel_pytree(tree)
    size = y.size
    key = jr.PRNGKey(18)
    matrix = jr.normal(key, (size, size))
    f = lambda x, _: matrix @ x
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1 / (2**4))
    return unflatten(_midpoint_f_general(f, y0, dt, y, args))


def _midpoint_k_linear(tree: PyTree[Array], args: Any):
    (y, unflatten) = jax.flatten_util.ravel_pytree(tree)
    size = y.size
    key = jr.PRNGKey(19)
    matrix = jr.normal(key, (size, size))
    f = lambda x, _: matrix @ x
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1 / (2**4))
    return unflatten(_midpoint_k_general(f, y0, dt, y, args))


def _midpoint_y_nonlinear(y: PyTree[Array], args: Any):
    size = _getsize(y)
    assert size == 3
    f = Robertson(0.04, 3e7, 1e4)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = 1e-8
    return _midpoint_y_general(f, y0, dt, y, args)


def _midpoint_f_nonlinear(y: PyTree[Array], args: Any):
    size = _getsize(y)
    assert size == 3
    f = Robertson(0.04, 3e7, 1e4)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = 1e-8
    return _midpoint_f_general(f, y0, dt, y, args)


def _midpoint_k_nonlinear(y: PyTree[Array], args: Any):
    size = _getsize(y)
    assert size == 3
    f = Robertson(0.04, 3e7, 1e4)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = 1e-8
    return _midpoint_k_general(f, y0, dt, y, args)


ones_pytree = ({"a": 0.5 * jnp.ones((3, 2, 4))}, 0.5 * jnp.ones(4))
ones_robertson = (jnp.ones(2), {"b": jnp.array(1.0)})
hundred = jnp.ones(100)
fixed_point_problem_init = (
    (
        optx.FixedPointProblem(_sin),
        ones_pytree,
    ),
    (
        optx.FixedPointProblem(_exponential),
        ones_pytree,
    ),
    (
        optx.FixedPointProblem(_nn),
        ones_pytree,
    ),
    (
        optx.FixedPointProblem(_nonlinear_heat_pde),
        hundred,
    ),
    (
        optx.FixedPointProblem(_midpoint_y_linear),
        ones_pytree,
    ),
    (
        optx.FixedPointProblem(_midpoint_f_linear),
        ones_pytree,
    ),
    (
        optx.FixedPointProblem(_midpoint_k_linear),
        ones_pytree,
    ),
    (optx.FixedPointProblem(_midpoint_y_nonlinear), ones_robertson),
    (
        optx.FixedPointProblem(_midpoint_f_nonlinear),
        ones_robertson,
    ),
    (
        optx.FixedPointProblem(_midpoint_k_nonlinear),
        ones_robertson,
    ),
)
