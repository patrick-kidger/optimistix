import functools as ft
import operator
import random
from typing import Any, Callable

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar


def getkey():
    return jr.PRNGKey(random.randint(0, 2**31 - 1))


def _shaped_allclose(x, y, **kwargs):
    if type(x) is not type(y):
        return False
    if isinstance(x, jnp.ndarray):  # pyright: ignore
        if jnp.issubdtype(x.dtype, jnp.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and jnp.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and jnp.all(x == y)
    elif isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.inexact):
            return (
                x.shape == y.shape
                and x.dtype == y.dtype
                and np.allclose(x, y, **kwargs)
            )
        else:
            return x.shape == y.shape and x.dtype == y.dtype and np.all(x == y)
    elif isinstance(x, jax.ShapeDtypeStruct):
        assert x.shape == y.shape and x.dtype == y.dtype
    else:
        return x == y


def shaped_allclose(x, y, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jtu.tree_structure(x) == jtu.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, **kwargs)
    return same_structure and jtu.tree_reduce(
        operator.and_, jtu.tree_map(allclose, x, y), True
    )


def finite_difference_jvp(fn, primals, tangents):
    out = fn(*primals)
    # Choose ε to trade-off truncation error and floating-point rounding error.
    max_leaves = [jnp.max(jnp.abs(p)) for p in jtu.tree_leaves(primals)] + [1]
    scale = jnp.max(jnp.stack(max_leaves))
    ε = np.sqrt(np.finfo(np.float64).eps) * scale
    primals_ε = (ω(primals) + ε * ω(tangents)).ω
    out_ε = fn(*primals_ε)
    tangents_out = jtu.tree_map(lambda x, y: (x - y) / ε, out_ε, out)
    return out, tangents_out


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


def _bowl(tree: PyTree[Array], args: Array):
    # Trivial quadratic bowl smoke test for convergence.
    (y, _) = jfu.ravel_pytree(tree)
    matrix = args
    return y.T @ matrix @ y


def _diagonal_quadratic_bowl(tree: PyTree[Array], args: PyTree[Array]):
    # A diagonal quadratic bowl smoke test for convergence.
    weight_vector = args
    return (ω(tree).call(jnp.square) * weight_vector**ω).ω


def _rosenbrock(tree: PyTree[Array], args: Scalar):
    # Wiki
    # least squares
    (y, _) = jfu.ravel_pytree(tree)
    const = args
    return (100 * (y[1:] - y[:-1]), const - y[:-1])


def _himmelblau(tree: PyTree[Array], args: Any):
    # Wiki
    (y, z) = tree
    const1, const2 = args
    term1 = ((ω(y).call(jnp.square) + z**ω - const1) ** 2).ω
    term2 = ((y**ω + ω(z).call(jnp.square) - const2) ** 2).ω
    return (term1**ω + term2**ω).ω


def _matyas(tree: PyTree[Array], args: Any):
    # Wiki
    (y, z) = tree
    const1, const2 = args
    term1 = (const1 * (ω(y).call(jnp.square) + ω(z).call(jnp.square))).ω
    term2 = (const2 * y**ω * z**ω).ω
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
    const1, const2, const3 = args
    term1 = ((const1 - y**ω + y**ω * z**ω) ** 2).ω
    term2 = ((const2 - y**ω + y**ω * ω(z).call(jnp.square)) ** 2).ω
    term3 = ((const3 - y**ω + y**ω * ω(z).call(lambda x: x**3)) ** 2).ω
    return (term1**ω + term2**ω + term3**ω).ω


def penalty_ii(tree: PyTree[Array], args: Any):
    # TUOS eqn 24
    # least squares
    (y, _) = jfu.ravel_pytree(tree)
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


def variably_dimensioned(tree: PyTree[Array], args: Any):
    # TUOS eqn 25
    # least squares
    (y, _) = jfu.ravel_pytree(tree)
    const = args
    increasing = jnp.arange(1, jnp.size(y) + 1)
    term1 = y - const
    term2 = jnp.sum(increasing * (y - const))
    term3 = term2**2
    return (term1, term2, term3)


def trigonometric(tree: PyTree[Array], args: Any):
    # TUOS eqn 26
    # least squares
    (y, _) = jfu.ravel_pytree(tree)
    const = args
    sumcos = jnp.sum(jnp.cos(y))
    increasing = jnp.arange(1, jnp.size(y) + 1, dtype=jnp.float32)
    return jnp.size(y) - sumcos + increasing * (const - jnp.cos(y)) - jnp.sin(y)


def simple_nn(model_dynamic: PyTree[Array], args: Any):
    # args is the activation functions in
    # the MLP.
    (model_static, data) = args
    model = eqx.combine(model_dynamic, model_static)
    key = jr.PRNGKey(17)
    model_key, data_key = jr.split(key, 2)
    x = jnp.linspace(0, 1, 100)[..., None]
    y = data**2

    def loss(model, x, y):
        pred_y = eqx.filter_vmap(model)(x)
        return (pred_y - y) ** 2

    return loss(model, x, y)


#
# The MLP can be difficult for some of the solvers to optimise. Rather than set
# max_steps to a higher value and iterate for longer, we initialise the MLP
# closer to the minimum by explicitly setting the weights and biases.
#


def get_weights(model):
    layer1, layer2 = model.layers
    return layer1.weight, layer1.bias, layer2.weight, layer2.bias


ffn_init = eqx.nn.MLP(in_size=1, out_size=1, width_size=8, depth=1, key=jr.PRNGKey(17))
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
new_ffn_init = eqx.tree_at(get_weights, ffn_init, (weight1, bias1, weight2, bias2))
ffn_dynamic, ffn_static = eqx.partition(new_ffn_init, eqx.is_array)

# TODO(raderj): patch or remove _penalty_ii
# (
#     _penalty_ii,
#     jnp.array(2.93660e-4),
#     0.5 * jnp.ones((2, 5)),
# ),
# (
#     _penalty_ii,
#     jnp.array(2.93660e-4),
#     (({"a": 0.5 * jnp.ones(4)}), 0.5 * jnp.ones(3), (0.5 * jnp.ones(3), ())),
# ),

# _uncoupled_simple default args
diagonal_bowl_init = ({"a": 0.05 * jnp.ones((2, 3, 3))}, (0.05 * jnp.ones(2)))
leaves, treedef = jtu.tree_flatten(diagonal_bowl_init)
key = jr.PRNGKey(17)
diagonal_bowl_args = treedef.unflatten(
    [jr.normal(key, leaf.shape, leaf.dtype) ** 2 for leaf in leaves]
)

# neural net args
ffn_data = jnp.linspace(0, 1, 100)[..., None]
ffn_args = (ffn_static, ffn_data)


least_squares_fn_minima_init_args = (
    (
        _diagonal_quadratic_bowl,
        jnp.array(0.0),
        diagonal_bowl_init,
        diagonal_bowl_args,
    ),
    (
        _rosenbrock,
        jnp.array(0.0),
        [jnp.array(1.5), jnp.array(1.5)],
        jnp.array(1.0),
    ),
    (
        _rosenbrock,
        jnp.array(0.0),
        (1.5 * jnp.ones((2, 4)), {"a": 1.5 * jnp.ones((2, 3, 2))}, ()),
        jnp.array(1.0),
    ),
    (simple_nn, jnp.array(0.0), ffn_dynamic, ffn_args),
    (
        variably_dimensioned,
        jnp.array(0.0),
        1 - jnp.arange(1, 11) / 10,
        jnp.array(1.0),
    ),
    (
        variably_dimensioned,
        jnp.array(0.0),
        (1 - jnp.arange(1, 7) / 10, {"a": (1 - jnp.arange(7, 11) / 10)}),
        jnp.array(1.0),
    ),
    (
        trigonometric,
        jnp.array(0.0),
        jnp.ones(70) / 70,
        jnp.array(1.0),
    ),
    (
        trigonometric,
        jnp.array(0.0),
        ((jnp.ones(40) / 70, (), {"a": jnp.ones(20) / 70}), jnp.ones(10) / 70),
        jnp.array(1.0),
    ),
)

bowl_init = ({"a": 0.05 * jnp.ones((2, 3, 3))}, (0.05 * jnp.ones(2)))
(flatten_bowl, _) = jfu.ravel_pytree(bowl_init)
key = jr.PRNGKey(17)
matrix = jr.normal(key, (flatten_bowl.size, flatten_bowl.size))
diagonal_bowl_args = matrix.T @ matrix

minimisation_fn_minima_init_args = (
    (_bowl, jnp.array(0.0), bowl_init, diagonal_bowl_args),
    (
        _himmelblau,
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(2.5)],
        (jnp.array(11.0), jnp.array(7.0)),
    ),
    (
        _matyas,
        jnp.array(0.0),
        [jnp.array(6.0), jnp.array(6.0)],
        (jnp.array(0.26), jnp.array(0.48)),
    ),
    (
        _beale,
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(0.0)],
        (jnp.array(1.5), jnp.array(2.25), jnp.array(2.625)),
    ),
)

# ROOT FIND/FIXED POINT PROBLEMS
#
# These are mostly derived from common problems arising in
# ordinary and partial differential equations. See
# Diffrax for more on differential equations:
# https://github.com/patrick-kidger/diffrax.
#
# - Simple smoke tests
# - MLP(y)
# - Nonlinear heat equation with Crank Nicholson
# - Implicit midpoint Runge-Kutta in y-space, f-space, and k-space
#


# SETUP
# Helper functions, most of which are single steps of a differential
# equation solve.
def _getsize(y: PyTree[Array]):
    return jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(jnp.size, y))


def _laplacian(y: PyTree[Array], dx: Scalar):
    (y, unflatten) = jfu.ravel_pytree(y)
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
    const = args
    stepsize = t1 - t0
    f_val = ((1 - y**ω) * _laplacian(y, dx) ** ω).ω
    return const * (y0**ω + 0.5 * stepsize * (f_val**ω + f0**ω)).ω


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
        (y, unflatten) = jfu.ravel_pytree(y)
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return unflatten(jnp.stack([f0, f1, f2]))


# PROBLEMS
def _exponential(tree: PyTree[Array], args: Any):
    const = args
    return jtu.tree_map(lambda x: jnp.exp(-const * x), tree)


def _sin(tree: PyTree[Array], args: Any):
    const = args
    return jtu.tree_map(lambda x: jnp.sin(const * x), tree)


def _nn(tree: PyTree[Array], args: Any):
    (y, unflatten) = jfu.ravel_pytree(tree)
    size = _getsize(y)
    weight1, bias1, weight2, bias2 = args
    model = eqx.nn.MLP(
        in_size=size,
        out_size=size,
        width_size=8,
        depth=1,
        activation=jax.nn.softplus,  # more complex than relu
        key=jr.PRNGKey(42),  # doesn't matter! is overwritten immediately
    )
    model = eqx.tree_at(get_weights, model, (weight1, bias1, weight2, bias2))
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
    (y, unflatten) = jfu.ravel_pytree(tree)
    matrix = args
    f = lambda x, _: matrix @ x
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1 / (2**4))
    midpoint = _midpoint_y_general(f, y0, dt, y, args)
    return unflatten(midpoint)


def _midpoint_f_linear(tree: PyTree[Array], args: Any):
    (y, unflatten) = jfu.ravel_pytree(tree)
    f = lambda x, args: jnp.dot(args, x)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1 / (2**4))
    midpoint = _midpoint_f_general(f, y0, dt, y, args)
    if midpoint.size == 1:
        out = midpoint.reshape(1)
    else:
        out = midpoint
    return unflatten(out)


def _midpoint_k_linear(tree: PyTree[Array], args: Any):
    (y, unflatten) = jfu.ravel_pytree(tree)
    f = lambda x, args: jnp.dot(args, x)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1 / (2**4))
    midpoint = _midpoint_k_general(f, y0, dt, y, args)
    if midpoint.size == 1:
        out = midpoint.reshape(1)
    else:
        out = midpoint
    return unflatten(out)


def _midpoint_y_nonlinear(y: PyTree[Array], args: Any):
    size = _getsize(y)
    assert size == 3
    const1, const2, const3 = args
    f = Robertson(const1, const2, const3)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1e-8)
    return _midpoint_y_general(f, y0, dt, y, args)


def _midpoint_f_nonlinear(y: PyTree[Array], args: Any):
    size = _getsize(y)
    assert size == 3
    const1, const2, const3 = args
    f = Robertson(const1, const2, const3)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1e-8)
    return _midpoint_f_general(f, y0, dt, y, args)


def _midpoint_k_nonlinear(y: PyTree[Array], args: Any):
    size = _getsize(y)
    assert size == 3
    const1, const2, const3 = args
    f = Robertson(const1, const2, const3)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1e-8)
    return _midpoint_k_general(f, y0, dt, y, args)


def trivial(y: Array, args: Any):
    return y - 1


# ARGS FOR BISECTION AND FIXED POINT
ones_pytree = ({"a": 0.5 * jnp.ones((3, 2, 4))}, 0.5 * jnp.ones(4))
flat_ones, _ = jfu.ravel_pytree(ones_pytree)
size = flat_ones.size
key = jr.PRNGKey(19)
matrix = jr.normal(key, (size, size))
nn_model = eqx.nn.MLP(
    in_size=size,
    out_size=size,
    width_size=8,
    depth=1,
    activation=jax.nn.softplus,  # more complex than relu
    key=jr.PRNGKey(42),
)
nn_args = get_weights(nn_model)
robertson_args = (jnp.array(0.04), jnp.array(3e7), jnp.array(1e4))
ones_robertson = (jnp.ones(2), {"b": jnp.array(1.0)})
hundred = jnp.ones(100)
single = jnp.array(1.0)
bisection_fn_init_options_args = (
    (
        trivial,
        single,
        {"upper": jnp.array(2.0), "lower": jnp.array(0.5)},
        None,
    ),
    (
        _sin,
        single,
        {"upper": jnp.array(1.0), "lower": jnp.array(0.0)},
        jnp.array(1.0),
    ),
    (
        _exponential,
        single,
        {"upper": jnp.array(1.0), "lower": jnp.array(0.0)},
        jnp.array(1.0),
    ),
    (
        _midpoint_y_linear,
        single,
        {"upper": jnp.array(1.0), "lower": jnp.array(0.0)},
        jnp.array([0.5]),
    ),
    (
        _midpoint_f_linear,
        single,
        {"upper": jnp.array(1.0), "lower": jnp.array(0.0)},
        jnp.array([0.625]),
    ),
    (
        _midpoint_k_linear,
        single,
        {"upper": jnp.array(1.0), "lower": jnp.array(0.0)},
        jnp.array([-0.137]),
    ),
)

fixed_point_fn_init_args = (
    (_sin, ones_pytree, jnp.array(1.0)),
    (_exponential, ones_pytree, jnp.array(1.0)),
    (_nn, ones_pytree, nn_args),
    (_nonlinear_heat_pde, hundred, jnp.array(1.0)),
    (
        _midpoint_y_linear,
        ones_pytree,
        matrix,
    ),
    (
        _midpoint_f_linear,
        ones_pytree,
        matrix,
    ),
    (
        _midpoint_k_linear,
        ones_pytree,
        matrix,
    ),
    (_midpoint_y_nonlinear, ones_robertson, robertson_args),
    (
        _midpoint_f_nonlinear,
        ones_robertson,
        robertson_args,
    ),
    (
        _midpoint_k_nonlinear,
        ones_robertson,
        robertson_args,
    ),
)
