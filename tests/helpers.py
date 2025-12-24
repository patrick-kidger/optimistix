import functools as ft
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import diffrax as dfx
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import numpy as np
import optax
import optimistix as optx
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar
from optimistix._misc import default_verbose, tree_full_like


Y = TypeVar("Y")
Out = TypeVar("Out")
Aux = TypeVar("Aux")


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def finite_difference_jvp(fn, primals, tangents, eps=None, **kwargs):
    assert jax.config.jax_enable_x64  # pyright: ignore
    out = fn(*primals, **kwargs)
    # Choose ε to trade-off truncation error and floating-point rounding error.
    max_leaves = [jnp.max(jnp.abs(p)) for p in jtu.tree_leaves(primals)] + [1]
    scale = jnp.max(jnp.stack(max_leaves))
    if eps is None:
        ε = np.sqrt(np.finfo(np.float64).eps) * scale
    else:
        # Sometimes we may want to set it manually. finite_difference_jvp is actually
        # pretty inaccurate for nonlinear solves, as these are themselves often only
        # done to a tolerance of 1e-8 or so: the primal pass is already noisy at about
        # the scale of ε.
        ε = eps
    with jax.numpy_dtype_promotion("standard"):
        primals_ε = (ω(primals) + ε * ω(tangents)).ω
        out_ε = fn(*primals_ε, **kwargs)
        tangents_out = jtu.tree_map(lambda x, y: (x - y) / ε, out_ε, out)
    # We actually return the perturbed primal.
    # This should still be within all tolerance checks, and means that we have aceesss
    # to both the true primal and the perturbed primal when debugging.
    return out_ε, tangents_out


#
# NOTE: `GN` is shorthand for `gauss_newton`. We want to be sure we test every
# branch of `GN=True` and `GN=False` for all of these solvers.
#
# UNCONSTRAINED MIN
#
# SOLVERS:
# - Gauss-Newton (LM)
# - BFGS
# - LBFGS
# - GradientDescent
# - Optax
# - NonlinearCG
#
# LINE SEARCHES:
# - BacktrackingArmijo
# - ClassicalTrustRegion
# - LearningRate
#
# DESCENTS:
# - DampedNewtonDescent(GN=...)
# - IndirectDampedNewtonDescent(GN=...)
# - DoglegDescent(GN=...)
# - NewtonDescent(GN=...)
# - Gradient (Not appropriate for GN!)
# - NonlinearCGDescent (Not appropriate for GN!)
#

#
# We define a bunch of helper optimisers which exist to test all code paths, but which
# are not default solvers because they feature methods only advanced users are likely
# to use.
#


class DoglegMax(optx.AbstractGaussNewton[Y, Out, Aux]):
    """Dogleg with trust region shape given by the max norm instead of the two norm."""

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: optx.DoglegDescent[Y]
    search: optx.ClassicalTrustRegion[Y]
    verbose: Callable[..., None]

    def __init__(
        self,
        rtol: float,
        atol: float,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = optx.max_norm
        self.descent = optx.DoglegDescent(
            linear_solver=lx.AutoLinearSolver(well_posed=False),
            root_finder=optx.Bisection(rtol=0.001, atol=0.001),
            trust_region_norm=optx.max_norm,
        )
        self.search = optx.ClassicalTrustRegion()
        self.verbose = default_verbose(False)


class BFGSDampedNewton(optx.AbstractBFGS):
    """BFGS Hessian + direct Levenberg Marquardt update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.DampedNewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


class BFGSIndirectDampedNewton(optx.AbstractBFGS):
    """BFGS Hessian + indirect Levenberg Marquardt update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.IndirectDampedNewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


class BFGSDogleg(optx.AbstractBFGS):
    """BFGS Hessian + dogleg update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.DoglegDescent(linear_solver=lx.SVD())
    verbose: Callable[..., None] = default_verbose(False)


class BFGSLinearTrustRegion(optx.AbstractBFGS):
    """Standard BFGS + linear trust region update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


class BFGSLinearTrustRegionHessian(optx.AbstractBFGS):
    """Standard BFGS (uses hessian, not inverse!) + linear trust region update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


class BFGSClassicalTrustRegionHessian(optx.AbstractBFGS):
    """Standard BFGS (uses hessian, not inverse!) + classical trust region update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


class DFPDampedNewton(optx.AbstractDFP):
    """DFP Hessian + direct Levenberg Marquardt update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.DampedNewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


class DFPIndirectDampedNewton(optx.AbstractDFP):
    """DFP Hessian + indirect Levenberg Marquardt update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.IndirectDampedNewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


class DFPDogleg(optx.AbstractDFP):
    """DFP Hessian + dogleg update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.DoglegDescent(linear_solver=lx.SVD())
    verbose: Callable[..., None] = default_verbose(False)


class DFPClassicalTrustRegionHessian(optx.AbstractDFP):
    """Standard DFP (uses hessian, not inverse!) + classical trust region update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: Callable[..., None] = default_verbose(False)


atol = rtol = 1e-8
_lsqr_only = (
    optx.LevenbergMarquardt(rtol, atol),
    optx.IndirectLevenbergMarquardt(rtol, atol),
    optx.GaussNewton(rtol, atol, linear_solver=lx.AutoLinearSolver(well_posed=False)),
    optx.Dogleg(rtol, atol, linear_solver=lx.AutoLinearSolver(well_posed=False)),
    DoglegMax(rtol, atol),
)


atol = rtol = 1e-8
_general_minimisers = (
    optx.NelderMead(rtol, atol),
    optx.BFGS(rtol, atol, use_inverse=False),
    optx.BFGS(rtol, atol, use_inverse=True),
    optx.LBFGS(rtol, atol, use_inverse=False),
    optx.LBFGS(rtol, atol, use_inverse=True),
    BFGSDampedNewton(rtol, atol),
    BFGSIndirectDampedNewton(rtol, atol),
    # Tighter tolerance needed to have BFGSDogleg pass the JVP test.
    BFGSDogleg(1e-10, 1e-10),
    optx.OptaxMinimiser(optax.adam(learning_rate=3e-3), rtol=rtol, atol=atol),
    # optax.lbfgs includes their linesearch by default
    optx.OptaxMinimiser(optax.lbfgs(), rtol=rtol, atol=atol),
)

_minim_only = (
    BFGSClassicalTrustRegionHessian(rtol, atol),
    BFGSLinearTrustRegionHessian(rtol, atol),
    BFGSLinearTrustRegion(rtol, atol),
    optx.DFP(rtol, atol, use_inverse=False),
    optx.DFP(rtol, atol, use_inverse=True),
    DFPDampedNewton(rtol, atol),
    DFPIndirectDampedNewton(rtol, atol),
    # Tighter tolerance needed to have DFPDogleg pass the JVP test.
    DFPDogleg(1e-10, 1e-10),
    DFPClassicalTrustRegionHessian(rtol, atol),
    optx.GradientDescent(1.5e-2, rtol, atol),
    # Tighter tolerance needed to have NonlinearCG pass the JVP test.
    optx.NonlinearCG(1e-10, 1e-10),
    # explicitly including a linesearch
    optx.OptaxMinimiser(
        optax.chain(
            optax.sgd(learning_rate=1.0),
            optax.scale_by_zoom_linesearch(15, curv_rtol=jnp.inf),
        ),
        rtol=rtol,
        atol=atol,
    ),
    optx.OptaxMinimiser(
        optax.chain(
            optax.sgd(learning_rate=1.0),
            optax.scale_by_backtracking_linesearch(15),
        ),
        rtol=rtol,
        atol=atol,
    ),
)

minimisers = _general_minimisers + _minim_only

# the minimisers can handle least squares problems, but the least squares
# solvers cannot handle general minimisation problems.
# without the ones that work, but are just pretty bad!
least_squares_optimisers = _lsqr_only + _general_minimisers


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


def bowl(tree: PyTree[Array], args: Array):
    # Trivial quadratic bowl smoke test for convergence.
    (y, _) = jfu.ravel_pytree(tree)
    matrix = args
    return y.T @ matrix @ y


def diagonal_quadratic_bowl(tree: PyTree[Array], args: PyTree[Array]):
    # A diagonal quadratic bowl smoke test for convergence.
    weight_vector = args
    return (ω(tree).call(jnp.square) * (0.1 + weight_vector**ω)).ω


def rosenbrock(tree: PyTree[Array], args: Scalar):
    # Wiki
    # least squares
    (y, _) = jfu.ravel_pytree(tree)
    const = args
    return (100 * (y[1:] - y[:-1]), const - y[:-1])


def _himmelblau(tree: PyTree[Array], args: PyTree):
    # Wiki
    (y, z) = tree
    const1, const2 = args
    term1 = ((ω(y).call(jnp.square) + z**ω - const1) ** 2).ω
    term2 = ((y**ω + ω(z).call(jnp.square) - const2) ** 2).ω
    return (term1**ω + term2**ω).ω


def matyas(tree: PyTree[Array], args: PyTree):
    # Wiki
    (y, z) = tree
    const1, const2 = args
    term1 = (const1 * (ω(y).call(jnp.square) + ω(z).call(jnp.square))).ω
    term2 = (const2 * y**ω * z**ω).ω
    return (term1**ω - term2**ω).ω


def _eggholder(tree: PyTree[Array], args: PyTree):
    # Wiki
    (y, z) = tree
    # the composition sin . sqrt . abs
    ssa = lambda x: jnp.sin(jnp.sqrt(jnp.abs(x)))
    term1 = (-(z**ω + 47.0) * (0.5 * y**ω + z**ω + 47.0).call(ssa)).ω
    term2 = (-(y**ω) * (y**ω - (z**ω - 47.0)).call(ssa)).ω
    return (term1**ω + term2**ω).ω


def beale(tree: PyTree[Array], args: PyTree):
    # Wiki
    (y, z) = tree
    const1, const2, const3 = args
    term1 = ((const1 - y**ω + y**ω * z**ω) ** 2).ω
    term2 = ((const2 - y**ω + y**ω * ω(z).call(jnp.square)) ** 2).ω
    term3 = ((const3 - y**ω + y**ω * ω(z).call(lambda x: x**3)) ** 2).ω
    return (term1**ω + term2**ω + term3**ω).ω


def variably_dimensioned(tree: PyTree[Array], args: PyTree):
    # TUOS eqn 25
    # least squares
    (y, _) = jfu.ravel_pytree(tree)
    const = args
    increasing = jnp.arange(1, jnp.size(y) + 1)
    term1 = y - const
    term2 = jnp.sum(increasing * (y - const))
    term3 = term2**2
    return (term1, term2, term3)


def trigonometric(tree: PyTree[Array], args: PyTree):
    # TUOS eqn 26
    # least squares
    (y, _) = jfu.ravel_pytree(tree)
    const = args
    sumcos = jnp.sum(jnp.cos(y))
    increasing = jnp.arange(1, jnp.size(y) + 1, dtype=jnp.float32)
    return jnp.size(y) - sumcos + increasing * (const - jnp.cos(y)) - jnp.sin(y)


def simple_nn(model_dynamic: PyTree[Array], args: PyTree):
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


def square_minus_one(x: Array, args: PyTree):
    """A simple ||x||^2 - 1 function."""
    return jnp.sum(jnp.square(x)) - 1.0


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
        diagonal_quadratic_bowl,
        jnp.array(0.0),
        diagonal_bowl_init,
        diagonal_bowl_args,
    ),
    (
        rosenbrock,
        jnp.array(0.0),
        [jnp.array(1.5), jnp.array(1.5)],
        jnp.array(1.0),
    ),
    (
        rosenbrock,
        jnp.array(0.0),
        (1.45 * jnp.ones((2, 4)), {"a": 1.45 * jnp.ones((2, 3, 2))}, ()),
        jnp.array(1.0),
    ),
    (simple_nn, jnp.array(0.0), ffn_dynamic, ffn_args),
    # Skipped for being hard!
    #
    # (
    #     variably_dimensioned,
    #     jnp.array(0.0),
    #     1 - jnp.arange(1, 11) / 10,
    #     jnp.array(1.0),
    # ),
    # (
    #     variably_dimensioned,
    #     jnp.array(0.0),
    #     (1 - jnp.arange(1, 7) / 10, {"a": (1 - jnp.arange(7, 11) / 10)}),
    #     jnp.array(1.0),
    # ),
    # (
    #     trigonometric,
    #     jnp.array(0.0),
    #     jnp.ones(70) / 70,
    #     jnp.array(1.0),
    # ),
    # (
    #     trigonometric,
    #     jnp.array(0.0),
    #     ((jnp.ones(40) / 70, (), {"a": jnp.ones(20) / 70}), jnp.ones(10) / 70),
    #     jnp.array(1.0),
    # ),
)

bowl_init = ({"a": 0.05 * jnp.ones((2, 3, 3))}, (0.05 * jnp.ones(2)))
(flatten_bowl, _) = jfu.ravel_pytree(bowl_init)
key = jr.PRNGKey(17)
matrix = jr.normal(key, (flatten_bowl.size, flatten_bowl.size))
diagonal_bowl_args = matrix.T @ matrix

minimisation_fn_minima_init_args = (
    (bowl, jnp.array(0.0), bowl_init, diagonal_bowl_args),
    (
        _himmelblau,
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(2.5)],
        (jnp.array(11.0), jnp.array(7.0)),
    ),
    (
        matyas,
        jnp.array(0.0),
        [jnp.array(6.0), jnp.array(6.0)],
        (jnp.array(0.26), jnp.array(0.48)),
    ),
    (
        beale,
        jnp.array(0.0),
        [jnp.array(2.0), jnp.array(0.0)],
        (jnp.array(1.5), jnp.array(2.25), jnp.array(2.625)),
    ),
    # Problems with initial value of 0
    (square_minus_one, jnp.array(-1.0), jnp.array(1.0), None),
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
    with jax.numpy_dtype_promotion("standard"):
        laplacian = laplacian.at[1:-1].set((y[2:] + y[1:-1] + y[:-2]) / dx)
    return unflatten(y)


def _nonlinear_heat_pde_general(
    y0: PyTree[Array],
    f0: PyTree[Array],
    dx: Scalar,
    t0: Scalar,
    t1: Scalar,
    y: PyTree[Array],
    args: PyTree,
):
    # A single step time `t0` to time `t1` of the Crank Nicolson scheme applied to
    # the nonlinear heat equation:
    # `d y(t, x)/dt = (1 - y(t, x)) Δy(t, x)`
    # where `d/dt` is a partial derivative wrt time and `Δ` is the Laplacian.
    const = args
    stepsize = t1 - t0
    f_val = ((1 - y**ω) * _laplacian(y, dx) ** ω).ω
    with jax.numpy_dtype_promotion("standard"):
        return const * (y0**ω + 0.5 * stepsize * (f_val**ω + f0**ω)).ω


# Note that the midpoint methods below assume that `f` is autonomous.


def _midpoint_y_general(
    f: Callable[[PyTree[Array], Any], PyTree[Array]],
    y0: PyTree[Array],
    dt: Scalar,
    y: PyTree[Array],
    args: PyTree,
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
    args: PyTree,
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
    args: PyTree,
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
def _exponential(tree: PyTree[Array], args: PyTree):
    const = args
    return jtu.tree_map(lambda x: jnp.exp(-const * x), tree)


def _cos(tree: PyTree[Array], args: PyTree):
    const = args
    return jtu.tree_map(lambda x: jnp.cos(const * x), tree)


def _nn(tree: PyTree[Array], args: PyTree):
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


def _nonlinear_heat_pde(y: PyTree[Array], args: PyTree):
    # solve the nonlinear heat equation as above from "0" to "1" in a
    # single step.
    x = jnp.linspace(-1, 1, 100)
    dx = x[1] - x[0]
    y0 = x**2
    f0 = ((1 - y0**ω) * _laplacian(y0, dx) ** ω).ω
    return _nonlinear_heat_pde_general(
        y0, f0, dx, jnp.array(0.0), jnp.array(1.0), y, args
    )


def _midpoint_y_linear(tree: PyTree[Array], args: PyTree):
    (y, unflatten) = jfu.ravel_pytree(tree)
    matrix = args
    f = lambda x, _: matrix @ x
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1 / (2**4))
    midpoint = _midpoint_y_general(f, y0, dt, y, args)
    return unflatten(midpoint)


def _midpoint_f_linear(tree: PyTree[Array], args: PyTree):
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


def _midpoint_k_linear(tree: PyTree[Array], args: PyTree):
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


def _midpoint_y_nonlinear(y: PyTree[Array], args: PyTree):
    size = _getsize(y)
    assert size == 3
    const1, const2, const3 = args
    f = Robertson(const1, const2, const3)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1e-8)
    return _midpoint_y_general(f, y0, dt, y, args)


def _midpoint_f_nonlinear(y: PyTree[Array], args: PyTree):
    size = _getsize(y)
    assert size == 3
    const1, const2, const3 = args
    f = Robertson(const1, const2, const3)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1e-8)
    return _midpoint_f_general(f, y0, dt, y, args)


def _midpoint_k_nonlinear(y: PyTree[Array], args: PyTree):
    size = _getsize(y)
    assert size == 3
    const1, const2, const3 = args
    f = Robertson(const1, const2, const3)
    y0 = ω(y).call(jnp.zeros_like).ω
    dt = jnp.array(1e-8)
    return _midpoint_k_general(f, y0, dt, y, args)


def trivial(y: Array, args: PyTree):
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
        _cos,
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
    (_cos, ones_pytree, jnp.array(1.0)),
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


# Not really useful enough to be part of the public API, but useful for testing against.
class PiggybackAdjoint(optx.AbstractAdjoint):
    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        del rewrite_fn, tags
        while_loop = ft.partial(eqxi.while_loop, kind="lax")
        return primal_fn(inputs + (while_loop,))


def forward_only_ode(k, args):
    # Test minimisers for use with dfx.ForwardMode. This test checks if the forward
    # branch is entered as expected and that a (trivial) result is found.
    # We're checking if trickier problems are solved correctly in the other tests.
    del args
    dy = lambda t, y, k: -k * y

    def solve(_k):
        return dfx.diffeqsolve(
            dfx.ODETerm(dy),
            dfx.Tsit5(),
            0.0,
            10.0,
            0.1,
            10.0,
            args=_k,
            adjoint=dfx.ForwardMode(),
        )

    data = jnp.asarray(solve(jnp.array(0.5)).ys)  # seems to make type checkers happy
    fit = jnp.asarray(solve(k).ys)
    return jnp.sum((data - fit) ** 2)


forward_only_fn_init_options_expected = (
    (
        forward_only_ode,
        jnp.array(0.6),
        dict(autodiff_mode="fwd"),
        jnp.array(0.5),
    ),
)


golden_search_fn_y0_options_expected = (
    (
        lambda y, args: (y - 2) ** 2,
        jnp.array(1),
        dict(lower=0, upper=3),
        jnp.array(2.0),
    ),
    # Edge case: minimum corresponds to lower bound
    (lambda y, args: y**2, jnp.array(1), dict(lower=0, upper=3), jnp.array(0.0)),
    # Edge case: minimum correspnds to upper bound
    (
        lambda y, args: (y - 2) ** 2,
        jnp.array(1),
        dict(lower=0, upper=2),
        jnp.array(2.0),
    ),
    # Function unbounded below, can be minimised within bounds
    (
        lambda y, args: -((y - 2) ** 2),
        jnp.array(1),
        dict(lower=0, upper=3),
        jnp.array(0.0),
    ),
)

# Define a bounded MLP (to check if clipping/projections works on complicated pytrees).
# Set weights and bias to known values to get a ground truth to test against
mlp = eqx.nn.MLP(2, 2, 2, 2, key=jr.key(0))
w1 = jnp.array([[1.0, -1.0], [0.5, 3.0]])
w2 = jnp.array([[4.0, 6.0], [-7, 8.0]])
w3 = jnp.array([[1.0, -2], [4, 5.0]])
b1 = jnp.array([[8.0, 1.0]])
b2 = jnp.array([[4.0, -2]])
b3 = jnp.array([[-3, 0.0]])
lw = jnp.zeros_like(w1)  # All weights and biases must be in (0, 1)
uw = jnp.ones_like(w1)
lb = jnp.zeros_like(b1)
ub = jnp.ones_like(1)

rw1 = jnp.clip(w1, min=lw, max=uw)
rw2 = jnp.clip(w2, min=lw, max=uw)
rw3 = jnp.clip(w3, min=lw, max=uw)
rb1 = jnp.clip(b1, min=lb, max=ub)
rb2 = jnp.clip(b2, min=lb, max=ub)
rb3 = jnp.clip(b3, min=lb, max=ub)


def get_nn_weights(model):
    l1, l2, l3 = model.layers
    return l1.weight, l1.bias, l2.weight, l2.bias, l3.weight, l3.bias


mlp_tree = eqx.tree_at(get_nn_weights, mlp, (w1, b1, w2, b2, w3, b3))
mlp_tree, _ = eqx.partition(mlp_tree, eqx.is_array)
mlp_lower = eqx.tree_at(get_nn_weights, mlp, (lw, lb, lw, lb, lw, lb))
mlp_lower, _ = eqx.partition(mlp_lower, eqx.is_array)
mlp_upper = eqx.tree_at(get_nn_weights, mlp, (uw, ub, uw, ub, uw, ub))
mlp_upper, _ = eqx.partition(mlp_upper, eqx.is_array)
mlp_result = eqx.tree_at(get_nn_weights, mlp, (rw1, rb1, rw2, rb2, rw3, rb3))
mlp_result, _ = eqx.partition(mlp_result, eqx.is_array)


trees_to_clip = (
    # tree, lower, upper, result
    (jnp.array(3.0), jnp.array(0.0), jnp.array(2.0), jnp.array(2.0)),
    (
        (jnp.array([4.0, 6.0]), jnp.array(8.0)),
        (jnp.array([0.0, 1.0]), jnp.array(9.0)),
        (jnp.array([3.0, 5.0]), jnp.array(10.0)),
        (jnp.array([3.0, 5.0]), jnp.array(9.0)),
    ),
    (mlp_tree, mlp_lower, mlp_upper, mlp_result),
    (
        {"a": jnp.array(4.0), "c": {"1": 0.5 * jnp.ones((8,)), "2": jnp.array(3.0)}},
        {"a": jnp.array(5.0), "c": {"1": 0.0 * jnp.ones((8,)), "2": -jnp.inf}},
        {"a": jnp.array(6.0), "c": {"1": 0.2 * jnp.ones((8,)), "2": jnp.array(2.0)}},
        {"a": jnp.array(5.0), "c": {"1": 0.2 * jnp.ones((8,)), "2": jnp.array(2.0)}},
    ),
)


_true_tree = (jnp.array(8.0), jnp.arange(1.0, 3.0), {"a": jnp.ones(2)})
_false_tree = tree_full_like(_true_tree, jnp.inf)

tree_where__pred_true_false_expected = (
    # pred is a scalar
    (True, _true_tree, _false_tree, _true_tree),
    # pred is a partial pre-fix
    (
        (True, False, True),
        _true_tree,
        _false_tree,
        (jnp.array(8.0), jnp.inf * jnp.ones(2), {"a": jnp.ones(2)}),
    ),
    # pred has the same structure as true, false
    (tree_full_like(_true_tree, False), _true_tree, _false_tree, _false_tree),
)


y_bounds_step_offset_expected = (
    (
        jnp.array(3.0),
        (jnp.array(0.0), jnp.array(5.0)),  # initial point inside bounds
        jnp.array(1.0),  # step inside bounds
        None,
        jnp.array(1.0),  # take full step
    ),
    (
        jnp.array(3),
        (jnp.array(3.0), jnp.array(5)),  # initial point on boundary
        jnp.array(4.0),  # step exceeds distance to upper bound
        None,
        jnp.array(0.5),  # take at most half a step
    ),
    (
        jnp.array(3),
        (jnp.array(3.0), jnp.array(5)),  # initial point on boundary
        -jnp.array(4.0),  # step would take us into disallowed region
        None,
        jnp.array(0.0),  # no movement allowed
    ),
    (
        jnp.array(3),
        (-jnp.array(jnp.inf), jnp.array(5)),  # initial point inside bounds
        jnp.array(-1.0),  # step inside bounds
        None,
        jnp.array(1.0),  # take full step
    ),
    (
        jnp.array(3),
        (jnp.array(-5), jnp.array(jnp.inf)),  # initial point inside bounds
        jnp.array(-4.0),  # step inside bounds
        None,
        jnp.array(1.0),  # take full step
    ),
    (
        jnp.array(3),
        (jnp.array(-1), jnp.array(jnp.inf)),  # initial point inside bounds, upper inf
        jnp.array(-8.0),  # step would take us into disallowed region (negative)
        None,
        jnp.array(0.5),  # take at most half a step
    ),
    (
        jnp.array(3),
        (-jnp.array(jnp.inf), jnp.array(5)),
        jnp.array(1.0),
        jnp.array(0.5),  # offset to adjust the step: enforces strict interior
        jnp.array(1.0),
    ),
    (
        jnp.array(3),
        (-jnp.array(jnp.inf), jnp.array(5)),
        jnp.array(4.0),
        jnp.array(0.5),
        jnp.array(0.25),
    ),
    (
        jnp.array(3),
        (-jnp.array(jnp.inf), jnp.array(3)),
        jnp.array(1.0),
        None,
        jnp.array(0.0),
    ),
    (
        jnp.array([1.0, 2.0]),
        (-jnp.array([jnp.inf, jnp.inf]), jnp.array([3.0, 3.0])),
        jnp.array([1.0, 2.0]),
        None,
        jnp.array([1.0, 0.5]),
    ),
    (
        (jnp.array(1), jnp.array(2)),  # simple pytree structure
        ((-jnp.array(jnp.inf), -jnp.array(jnp.inf)), (jnp.array(3), jnp.array(4))),
        (jnp.array(1), jnp.array(2)),
        None,
        (jnp.array(1.0), jnp.array(1.0)),
    ),
    (
        jnp.array(-0.5),  # initial point outside bounds
        (jnp.array(0.0), jnp.array(5.0)),
        jnp.array(1.0),  # step takes us back into bounded region
        None,
        jnp.array(1.0),
    ),
    (
        jnp.array(-0.5),  # initial point outside bounds
        (jnp.array(0.0), jnp.array(5.0)),
        jnp.array(-1.0),  # step takes us further outside bounded region
        None,
        jnp.array(0.0),
    ),
    # No tree reduction over array leaves
    (
        (jnp.array(8.0), jnp.array([2.0, 4.0, 5.0])),
        ((-jnp.inf, jnp.arange(1.0, 4.0)), (jnp.array(9.0), jnp.arange(3.0, 6.0))),
        (jnp.array(0.1), jnp.ones(3)),
        None,
        (jnp.array(1.0), jnp.array([1.0, 0.0, 0.0])),
    ),
    (
        jnp.array([1.0, 0.0]),
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        jnp.array([1.0, -1.0]),
        None,
        jnp.array([0.0, 0.0]),
    ),
    (
        jnp.array([1.0, 0.0]),
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        jnp.array([1.0, -1.0]),
        0.5,
        jnp.array([0.0, 0.0]),
    ),
    (
        jnp.array([0.5, 0.5]),
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        jnp.array([1.0, -1.0]),
        None,
        jnp.array([0.5, 0.5]),
    ),
)


# Easy tests for bounded and constrained optimisation. These are all smoke tests, but
# they are varied enough that they help catch quite a few different cases early.
# For example, initial points may be on the constraint boundary, or be a (general)
# Cauchy point, upper as well as lower bounds may be blocking, etc.


def _paraboloid(y, args):
    del args
    squares = jtu.tree_map(lambda x: x**2, y)
    squares, _ = jfu.ravel_pytree(squares)
    return jnp.sum(squares)


class _Point(NamedTuple):
    a: float
    b: float


# Vary cases and pytree types: smoke tests with trivial quadratic function
# What do we mean by Cauchy point? This point is the first local minimiser
# of a piecewise linear path along the surface of a hypercube defined by the
# bounds on the optimisation problem, identified by solving for the minimum
# of the quadratic approximation to the target function on each of the segments
# making up the piecewise linear path. This point is used e.g. in BFGS-B and
# its variants to identify the set of active bound constraints, and then solve
# for a new direction in the unconstrained subspace.
# In the literature, it is sometimes also called the "generalised Cauchy point"
# (e.g. in Trust Region Methods by Conn, Gould, Toint).
# Below, if locations of the Cauchy point are indicated, these are given with
# respect to the initial point. For example, if the Cauchy point is at the
# minimum, then we expect to encounter it in the first step due to the projection
# onto the hypercube, even if a full gradient or Newton step would have led us
# to leave the feasible set.
bounded_paraboloids = (
    # fn, y0, args, bounds, expected result
    # No bounds active at (0.0, 0.0), bounds far from minimum
    (
        _paraboloid,
        jnp.array([-1.0, -5.0]),
        None,
        (jnp.array([-jnp.inf, -jnp.inf]), jnp.array([1.0, 1.0])),
        jnp.array([0.0, 0.0]),
    ),
    # One upper bound active at (0.0, 0.0), Cauchy point is at minimum
    (
        _paraboloid,
        jnp.array([-4.0, -1.0]),
        None,
        (jnp.array([-jnp.inf, -jnp.inf]), jnp.array([0.0, 1.0])),
        jnp.array([0.0, 0.0]),
    ),
    # Two upper bounds active at (0.0, 0.0), Cauchy point is at minimum
    (
        _paraboloid,
        [-1.0, -1.0],
        None,
        ([-jnp.inf, -jnp.inf], [0.0, 0.0]),
        [0.0, 0.0],
    ),
    # One bound active at (-1.0, 0.0)
    (
        _paraboloid,
        {"a": -3.0, "b": -1.0},
        None,
        ({"a": -jnp.inf, "b": -jnp.inf}, {"a": -1.0, "b": 1.0}),
        {"a": -1.0, "b": 0.0},
    ),
    # Two bounds active at (-1.0, 0.0), initial point at minimum and Cauchy point
    (
        _paraboloid,
        (-1.0, 0.0),
        None,
        ((-jnp.inf, -jnp.inf), (-1.0, 0.0)),
        (-1.0, 0.0),
    ),
    # One bound active at (0.0, -1.0), initial point out of bounds
    (
        _paraboloid,
        (0.0, {"b": -2.0}),
        None,
        ((-jnp.inf, {"b": -jnp.inf}), (1.0, {"b": -1.0})),
        (0.0, {"b": -1.0}),
    ),
    # Two bounds active at (0.0, -1.0)
    (
        _paraboloid,
        _Point(-1.0, -7.0),
        None,
        (_Point(-jnp.inf, -jnp.inf), _Point(0.0, -1.0)),
        _Point(0.0, -1.0),
    ),
    # Two bounds active at (-1.0, -1.)
    (
        _paraboloid,
        jnp.array([-3.0, -1.0]),
        None,
        (jnp.array([-jnp.inf, -jnp.inf]), jnp.array([-1.0, -1.0])),
        jnp.array([-1.0, -1.0]),
    ),
    # Two bounds active at (1, 1), lower bound blocking
    (
        _paraboloid,
        jnp.array([2.0, 3.0]),
        None,
        (jnp.array([1.0, 1.0]), jnp.array([jnp.inf, jnp.inf])),
        jnp.array([1.0, 1.0]),
    ),
)


def _scalar_rosenbrock(y, args):
    out = rosenbrock(y, args)
    flat, _ = jfu.ravel_pytree(out)
    return jnp.sum(flat)


minimise_bounded_with_local_minima = (
    # fn, y0, args, bounds, expected result
    (
        _himmelblau,
        [4.0, 1.0],  # Initialise between two minima
        (jnp.array(11.0), jnp.array(7.0)),
        ([0.0, 0.0], [5.0, 5.0]),  # Quadrant: I
        [3.0, 2.0],
    ),
    (
        _himmelblau,
        jnp.array([4.0, -1.0]),
        (jnp.array(11.0), jnp.array(7.0)),
        (jnp.array([0.0, -5.0]), jnp.array([5.0, 0.0])),  # II
        jnp.array([3.584428, -1.848126]),
    ),
    (
        _himmelblau,
        (-3.0, -2.0),  # TODO: This problem is sensitive to initialisation
        # This makes it a good test case for inertia correction and initialisation of
        # dual variables. It converges to a stationary point when started at (-3, -1).
        (jnp.array(11.0), jnp.array(7.0)),
        ((-5.0, -5.0), (0.0, 0.0)),  # Quadrant: III
        (-3.779310, -3.283186),
    ),
    (
        _himmelblau,
        [jnp.array(-3.0), jnp.array(2.0)],
        (jnp.array(11.0), jnp.array(7.0)),
        ([jnp.array(-8.0), jnp.array(0.0)], [jnp.array(0.0), jnp.array(8.0)]),  # IV
        [jnp.array(-2.805118), jnp.array(3.131312)],
    ),
    (
        _scalar_rosenbrock,
        (0.4, 0.0),
        None,
        ((-5.0, -6.0), (4.0, 5.0)),
        (1.0, 1.0),
    ),
)


# Cauchy point tests: these all use a simple paraboloid centered around the origin as
# the objective function. This means that the gradient evaluated at any of the
# following test points is equal to the reflection of this point about the origin, and
# the bounds determine where the Cauchy point then lies.
def _wrapped_paraboloid(y):
    return _paraboloid(y, None)


y00 = jnp.array([3.0, 3.0])
y01 = jnp.array([-2.0, 2.0])
y02 = jnp.array([2.0, 2.0])
y03 = jnp.array([1.0, 1.0])
y04 = jnp.array([3.0, 3.0])
y05 = jnp.array([0.0, 0.0])
y06 = jnp.array([2.0, 1.0])
y07 = jnp.array([3.0, 3.0])
y08 = jnp.array([3.0, 3.0])
y09 = jnp.array([0.0, 0.0])
y10 = jnp.array([3.0, 2.0])
y11 = jnp.array([3.0, 2.0])
y12 = jnp.array([-1.0, -1.0])
y13 = jnp.array([-2.0, -2.0])
y14 = jnp.array([1.0, 1.0])
y15 = jnp.array([1.0, 1.0])
y16 = jnp.array(1.0)


cauchy_point__y_bounds_grad_hessian_expected = (
    (
        y00,  # Cauchy point at lower bounds, shorter than full gradient step
        (jnp.array([-2.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y00),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y00),
            output_structure=jax.eval_shape(lambda: y00),
        ),
        jnp.array([-2.0, -2.0]),
    ),
    (
        y01,  # Full gradient step possible, lower blocking bound changes
        (jnp.array([-2.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y01),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y01),
            output_structure=jax.eval_shape(lambda: y01),
        ),
        jnp.array([2.0, -2.0]),
    ),
    (
        y02,  # Full gradient step coincides exactly with lower bounds
        (jnp.array([-2.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y02),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y02),
            output_structure=jax.eval_shape(lambda: y02),
        ),
        jnp.array([-2.0, -2.0]),
    ),
    (
        y03,  # Full gradient step possible, starts and ends inside feasible set
        (jnp.array([-2.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y03),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y03),
            output_structure=jax.eval_shape(lambda: y03),
        ),
        jnp.array([-1.0, -1.0]),
    ),
    (
        y04,  # ????
        (jnp.array([1.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y04),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y04),
            output_structure=jax.eval_shape(lambda: y04),
        ),
        jnp.array([1.0, 1.0]),
    ),
    (
        y05,  # Gradient is zero, no displacement
        (jnp.array([-2.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y05),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y05),
            output_structure=jax.eval_shape(lambda: y05),
        ),
        jnp.array([0.0, 0.0]),
    ),
    (
        y06,  # Lower bound blocking for y1, not blocking for y2
        (jnp.array([1.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y06),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y06),
            output_structure=jax.eval_shape(lambda: y06),
        ),
        jnp.array([1.0, -1.0]),
    ),
    (
        y07,  # Cauchy point at both lower bounds, nonfinite upper bounds
        (jnp.array([-2.0, -2.0]), jnp.array([jnp.inf, jnp.inf])),
        jax.grad(_wrapped_paraboloid)(y07),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y07),
            output_structure=jax.eval_shape(lambda: y07),
        ),
        jnp.array([-2.0, -2.0]),
    ),
    (
        y08,  # Full gradient step possible, nonfinite lower bounds not blocking
        (jnp.array([-jnp.inf, -jnp.inf]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y08),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y08),
            output_structure=jax.eval_shape(lambda: y08),
        ),
        jnp.array([-3.0, -3.0]),
    ),
    (
        y09,  # Zero gradient + starting point at lower bounds, no displacement
        (jnp.array([0.0, -jnp.inf]), jnp.array([3.0, jnp.inf])),
        jax.grad(_wrapped_paraboloid)(y09),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y09),
            output_structure=jax.eval_shape(lambda: y09),
        ),
        jnp.array([0.0, 0.0]),
    ),
    (
        y10,  # Full gradient step coincides with one lower bound
        (jnp.array([-jnp.inf, -2.0]), jnp.array([jnp.inf, 3.0])),
        jax.grad(_wrapped_paraboloid)(y10),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y10),
            output_structure=jax.eval_shape(lambda: y10),
        ),
        jnp.array([-3.0, -2.0]),
    ),
    (
        y11,  # Lower bound blocks y2, but not y1
        (jnp.array([1.0, 1.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y11),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y11),
            output_structure=jax.eval_shape(lambda: y11),
        ),
        jnp.array([1.5, 1.0]),
    ),
    (
        y12,  # Full gradient step possible (direction flipped w.r.t. earlier case)
        (jnp.array([-2.0, -2.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y12),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y12),
            output_structure=jax.eval_shape(lambda: y12),
        ),
        jnp.array([1.0, 1.0]),
    ),
    (
        y13,  # Satring point at lower bounds, upper bounds block full gradient step
        (jnp.array([-2.0, -2.0]), jnp.array([-0.5, 3.0])),
        jax.grad(_wrapped_paraboloid)(y13),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y13),
            output_structure=jax.eval_shape(lambda: y13),
        ),
        jnp.array([-0.5, -0.5]),
    ),
    (
        y14,  # Lower bound blocking, gradient pointing outside the feasible set
        (jnp.array([1.0, 1.0]), jnp.array([3.0, 3.0])),
        jax.grad(_wrapped_paraboloid)(y14),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y14),
            output_structure=jax.eval_shape(lambda: y14),
        ),
        jnp.array([1.0, 1.0]),
    ),
    (
        y15,  # Lower bound blocking, gradient pointing outside, w/ nonfinite bounds
        (jnp.array([-jnp.inf, 1.0]), jnp.array([jnp.inf, 3.0])),
        jax.grad(_wrapped_paraboloid)(y15),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y15),
            output_structure=jax.eval_shape(lambda: y15),
        ),
        jnp.array([1.0, 1.0]),
    ),
    (
        y16,  # 1D, lower bound blocking
        (jnp.array(0.0), jnp.array(2.0)),
        jax.grad(_wrapped_paraboloid)(y16),
        lx.PyTreeLinearOperator(
            jax.hessian(_wrapped_paraboloid)(y16),
            output_structure=jax.eval_shape(lambda: y16),
        ),
        jnp.array(0.0),
    ),
)
