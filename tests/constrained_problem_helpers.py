from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx

from .helpers import himmelblau, scalar_rosenbrock


class BFGSInterior(optx.AbstractOldBFGS):
    """BFGS Hessian + interior point update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.BacktrackingArmijo()
    descent: optx.AbstractDescent = optx.InteriorDescent()
    verbose: frozenset[str] = frozenset()


class BFGSInteriorLearningRate(optx.AbstractOldBFGS):
    """BFGS Hessian + interior point update + learning rate."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.LearningRate(1.0)
    descent: optx.AbstractDescent = optx.InteriorDescent()
    verbose: frozenset[str] = frozenset()


class BFGSInteriorFiltered(optx.AbstractOldBFGS):
    """BFGS Hessian + interior point update + filtered line search."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    # TODO(jhaffner): once we settle on how to specify the buffer size, change this here
    search: optx.AbstractSearch = optx.IPOPTLikeFilteredLineSearch(2**9)
    descent: optx.AbstractDescent = optx.InteriorDescent()
    verbose: frozenset[str] = frozenset()


# TODO: the constrained minimisers still seem to work?
convex_constrained_minimisers = (
    # optx.SLSQP(rtol=1e-3, atol=1e-6),
    BFGSInterior(rtol=1e-3, atol=1e-6),
    BFGSInteriorLearningRate(rtol=1e-3, atol=1e-6),
)
nonconvex_constrained_minimisers = (
    # BFGSInteriorFiltered(rtol=1e-3, atol=1e-6),
    # TODO: does this work with the new Quasi Newton BFGS?
    optx.IPOPTLike(rtol=0.0, atol=1e-9),  # Currently no role for atol here!
)


class InteriorPointBacktracking(optx.AbstractQuadraticSolver):
    """Interior point with backtracking line search."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    search: optx.AbstractSearch = optx.BacktrackingArmijo()
    descent: optx.AbstractDescent = optx.InteriorDescent()


# TODO work out what tolerances to apply to the lot of them
constrained_quadratic_solvers = (
    optx.InteriorPoint(rtol=1e-6, atol=1e-12),
    InteriorPointBacktracking(rtol=1e-6, atol=1e-12),
)


def make_residuals(fun, x, y):
    """Create a test case for clipping in Gauss-Newton-type solvers."""

    def residuals(params, args):
        del args
        return fun(x, params) - y

    return residuals


def many_minima(x, params):
    a, b = params
    return jnp.cos(x - a) + jnp.cos(x - b)


x = jnp.linspace(0.0, 10.0, 100)
y = many_minima(x, jnp.array([1.0, -2.0]))
data = y + jr.normal(jr.key(0), shape=y.shape)
_many_minima_residuals = make_residuals(many_minima, x, y)


def parabola(x, params):
    offset, scale = params
    return scale * (x - offset) ** 2


x = jnp.linspace(-5.0, 5.0, 100)
y = parabola(x, jnp.array([2.0, 3.0]))
data = y + jr.normal(jr.key(0), shape=y.shape)
_parabola_residuals = make_residuals(parabola, x, y)

least_squares_bounded_many_minima = (
    # fn, y0, args, bounds
    # For many minima: initialise in the vicinity of the minimum (and the feasible set)
    # solver otherwise runs out of steps. Therefore this is only a smoke test.
    (
        _many_minima_residuals,
        jnp.array([-1.0, 0.8]),  # Minimum at -2., 1.
        None,
        (jnp.array([-5.0, 0.0]), jnp.array([0.0, 5.0])),  # a negative, b positive
    ),
    (
        _many_minima_residuals,
        jnp.array([0.2, -3.5]),  # Minimum at 1., -2.
        None,
        (jnp.array([0.0, -5.0]), jnp.array([5.0, 0.0])),  # a positive, b negative
    ),
    # Test case with minimum outside allowed region, to verify that clipping occurs
    (
        _parabola_residuals,
        jnp.array([0.0, 0.0]),  # Minimum at 2., 3.
        None,
        (jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
    ),
)


# Easy tests for bounded and constrained optimisation. These are all smoke tests, but
# they are varied enough that they help catch quite a few different cases early.
# For example, initial points may be on the constraint boundary, or be a (general)
# Cauchy point, upper as well as lower bounds may be blocking, etc.
# They are all 2D, so they can easily be plotted to examine if anything is going wrong.


def _paraboloid(y, args):
    del args
    squares = jtu.tree_map(lambda x: x**2, y)
    squares, _ = jfu.ravel_pytree(squares)
    return jnp.sum(squares)


_Point = NamedTuple("_Point", [("a", float), ("b", float)])

# Vary cases and pytree types: smoke tests with trivial quadratic function
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


def _161(y, args):  # Nocedal & Wright (2006), Numerical Optimisation, problem 16.1
    # Test case with linearly dependent constraints, solution far from constraint bounds
    del args
    y1, y2 = y
    return 2 * y1 + 3 * y2 + 4 * y1**2 + 2 * y1 * y2 + y2**2


def _constraints_161(y):
    y1, y2 = y
    c1 = y1 - y2
    c2 = 4 - y1 - y2
    c3 = 3 - y1
    return None, jnp.array([c1, c2, c3])


def _163(y, args):  # Nocedal & Wright (2006), Numerical Optimisation, problem 16.3
    # Test case with linearly dependent constraints, solution in pointy vertex.
    # The value of f changes slowly along the constraint boundary, so the solver tends
    # to terminate early.
    del args
    y1, y2 = y
    return y1**2 + 2 * y2**2 - 2 * y1 - 6 * y1 - 2 * y1 * y2


def _constraint_163(y):
    y1, y2 = y
    c1 = 1 - 0.5 * (y1 + y2)
    c2 = 2 + y1 - 2 * y2
    c3 = y1  # TODO: express these as bounds instead
    c4 = y2
    return None, jnp.stack([c1, c2, c3, c4])


def _167(y, args):  # Nocedal & Wright (2006), problem 16.7
    del args
    x1, x2 = y
    return -(6 * x1 + 4 * x2 - 13 - x1**2 - x2**2)


def _constraint_exercise_167(y):
    x1, x2 = y
    c1 = 3 - x1 - x2
    c2 = x1
    c3 = x2
    return None, jnp.stack([c1, c2, c3])


# TODO: mark these as inequality tests
# Vary constraints and directions: smoke tests with paraboloid
convex_constrained_paraboloids = (
    # fn, y, args, constraint, expected_result
    # Constraint active at optimum, but not blocking
    (
        _paraboloid,
        {"a": 0.0, "b": -1.0},
        None,
        lambda y: (None, y["a"] - y["b"]),  # x > y >= 0.
        {"a": 0.0, "b": 0.0},
    ),
    # Constraint inactive at optimum
    (
        _paraboloid,
        (1.0, -1.0),
        None,
        lambda y: (None, 1 + y[0] - y[1]),  # 1 + x - y >= 0.
        (0.0, 0.0),
    ),
    # Constraint active at shifted optimum (blocking)
    (
        _paraboloid,
        jnp.array([1.5, 0.0]),
        None,
        lambda y: (None, -1 + y[0] - y[1]),  # -1 + x - y >= 0.
        jnp.array([0.5, -0.5]),
    ),
    # Different non-blocking active constraint, now scaling one variable
    (
        _paraboloid,
        [1.0, -1.8],
        None,
        lambda y: (None, y[0] - 3 * y[1]),  # x - 3 * y >= 0.
        [0.0, 0.0],
    ),
    # Constraint inactive at optimum
    (
        _paraboloid,
        _Point(1.0, 0.0),
        None,
        lambda y: (None, 2 + y.a - 3 * y.b),  # 2 + x - 3 * y >= 0.
        _Point(0.0, 0.0),
    ),
    # Slightly more complex constraint active at shifted optimum
    (
        _paraboloid,
        (1.0, jnp.array([-1.0])),
        None,
        lambda y: (None, -2 + y[0] - 3 * y[1]),  # -2 + x - 3 * y >= 0.
        (0.2, jnp.array([-0.6])),
    ),
    # Constraint active at optimum, different half-space infeasible
    (
        _paraboloid,
        jnp.array([-1.0, 0.0]),
        None,
        lambda y: (None, y[1] - y[0]),  # x <= y
        jnp.array([0.0, 0.0]),
    ),
    # Several constraints, inactive at optimum, constraints linearly dependent
    (
        _161,
        jnp.array([2.0, -4]),
        None,
        _constraints_161,
        jnp.array([1.667e-01, -1.667]),  # scipy result (cobyqa)
    ),
    # Several constraints, linearly dependent, two active at optimum
    (
        _163,
        jnp.array([0.1, 0.2]),
        None,
        _constraint_163,
        jnp.array([2.0, 0.0]),
    ),
    # Several constraints, linearly dependent, one active at optimum
    (
        _167,
        jnp.array([0.1, 2.8]),
        None,
        _constraint_exercise_167,
        jnp.array([2.0, 1.0]),
    ),
    # Nonlinear sine-shaped constraint, active at optimum, trajectory needs to go around
    # the minimum of the sine function from the starting point, so the search space is
    # not convex.
    (
        _paraboloid,
        jnp.array([-2.0, -1.0]),
        None,
        lambda y: (None, jnp.sin(y[0]) - y[1]),  # y <= sin(x)
        jnp.array([0.0, 0.0]),
    ),
)


minimise_bounded_with_local_minima = (
    # fn, y0, args, bounds, expected result
    (
        himmelblau,
        [4.0, 1.0],  # Initialise between two minima
        (jnp.array(11.0), jnp.array(7.0)),
        ([0.0, 0.0], [5.0, 5.0]),  # Quadrant: I
        [3.0, 2.0],
    ),
    (
        himmelblau,
        jnp.array([4.0, -1.0]),
        (jnp.array(11.0), jnp.array(7.0)),
        (jnp.array([0.0, -5.0]), jnp.array([5.0, 0.0])),  # II
        jnp.array([3.584428, -1.848126]),
    ),
    (
        himmelblau,
        (-3.0, -2.0),  # TODO: This problem is sensitive to initialisation
        # This makes it a good test case for inertia correction and initialisation of
        # dual variables. It converges to a stationary point when started at (-3, -1).
        (jnp.array(11.0), jnp.array(7.0)),
        ((-5.0, -5.0), (0.0, 0.0)),  # Quadrant: III
        (-3.779310, -3.283186),
    ),
    (
        himmelblau,
        [jnp.array(-3.0), jnp.array(2.0)],
        (jnp.array(11.0), jnp.array(7.0)),
        ([jnp.array(-8.0), jnp.array(0.0)], [jnp.array(0.0), jnp.array(8.0)]),  # IV
        [jnp.array(-2.805118), jnp.array(3.131312)],
    ),
    (
        scalar_rosenbrock,
        (0.4, 0.0),
        None,
        ((-5.0, -6.0), (4.0, 5.0)),
        (1.0, 1.0),
    ),
)


# TODO(jhaffner): Consider what I want to test with these. The first one is smoke-y, the
# other ones can be tweaked to get more interesting properties of the solver.
minimise_fn_y0_args_constraint_expected_result = (
    (
        scalar_rosenbrock,
        (1.0, 0.0),
        None,
        lambda y: (2 - y[1], None),  # y == 2
        (1.41369616, 2.0),
    ),
    # Both variables are determined by equality constraints - check that we still do the
    # right thing. Does not currently check the number of iterations, though!
    (
        scalar_rosenbrock,
        (1.0, 0.0),
        None,
        lambda y: (jnp.array([1.2 - y[1], 1.2 - y[0]]), None),  # y = x = 1.2
        (1.2, 1.2),
    ),
    # TODO: this is a good problem to validate the inertia correction on. Both our IPOPT
    # and the scipy trust-constr interior point solver converge to a stationary point
    # of the Lagrangian when started at the origin. (This did not happen with quasi-
    # Newton Hessian approximations, but we currently don't use them).
    # Nonlinear constraint, convex in search region.
    # TODO: start at origin should go into the tricky geometries test cases
    (
        scalar_rosenbrock,
        (1.0, 0.0),
        None,
        lambda y: (jnp.sin(y[0]) + 0.2 - y[1], None),  # y <= sin(x) + 0.2
        (1.02754803, 1.05603404),
    ),
    # TODO: Currently does not manage to leave the local minimum if started at the
    # origin. Good test case to validate second order corrections, since the line
    # defined by the equality constraint goes through the local and global minimum.
    # TODO: add this - with start at origin - to the tricky geometries test cases
    (
        scalar_rosenbrock,
        (2.0, 0.0),
        None,
        lambda y: (
            y[0] - y[1],
            None,
        ),  # y <= x + 0.01  # TODO fix this for small offset
        (1.0, 1.0),
    ),
    (
        scalar_rosenbrock,
        (1.0, 0.0),
        None,
        lambda y: (y[0] ** 2 + 0.1 - y[1], None),
        (1.0, 1.1),
    ),
)


#
# Test cases for convex boundary maps
#


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

# TODO: define expected results for boundary maps other than box projection


# Test cases for nonconvex boundary maps, where the constraints can be nonlinear
outside_nonconvex_set__y_constraint_bounds_expected_result = (
    # y, constraint, bounds, expected_result
    # Sine shaped boundary of feasible set
    (
        jnp.array([5.0, 1.0]),
        lambda y: (None, jnp.sin(y[0]) - y[1]),  # y <= sin(x)
        None,
        jnp.array([5.8170223, -0.44946206]),
    ),
    # Smoke test with linear constraint
    (
        (1.0, 2.0),
        lambda y: (None, y[0] - y[1]),  # x >= y
        None,
        (1.2, 1.2),
    ),
)


# Problems for IPOPT
# Solvers that compute steps that satisfy linearized constraints and employ a line
# search without any mechanism to trade off improvement in the objective and improvement
# in the feasibility of the problem can fail if the feasible set is not defined by a
# smooth manifold. If there are inaccesible regions, for example due to bounds, then
# solvers of this type cannot converge to the solution depending on the starting point.
# For a more in-depth discussion of this example, see sections 3.3.3 and 4.1 of this
# thesis: https://users.iems.northwestern.edu/~andreasw/pubs/waechter_thesis.pdf
# Note that they say that this tests the "filtered line search", but by that they mean
# the whole algorithm - solving these problems hinges on being able to change direction,
# not just on being able to accept steps that improve the constraint violation without
# improving the objective function. Changed directions can come from second-order
# corrections, or from a feasibility restoration phase.


def inaccessible_region(y, args):
    del args
    x1, *_ = y
    return x1


def constraint_inaccessible_region(y):
    x1, x2, x3 = y
    c1 = x1**2 - x2 - 1
    c2 = x1 - x3 - 0.5
    return [c1, c2], None  # equality constraint is simple pytree


tricky_geometries__fn_y0_args_constraint_bounds_expected_result = (
    # fn, y0, args, constraint, bounds, expected_result
    (
        inaccessible_region,
        jnp.array([-5, 0.5, 0.51]),
        None,
        constraint_inaccessible_region,
        (jnp.array([-jnp.inf, 0, 0]), jnp.array([jnp.inf, jnp.inf, jnp.inf])),
        jnp.array([1.0, 0.0, 0.5]),
    ),
)


y__bounds__step__offset__expected_result = (
    (
        jnp.array(3.0),
        (jnp.array(0.0), jnp.array(5.0)),
        jnp.array(1.0),
        None,
        jnp.array(1.0),
    ),
    (
        jnp.array(3),
        (jnp.array(3.0), jnp.array(5)),  # initial point on boundary
        jnp.array(4.0),  # step toward allowed region
        None,
        jnp.array(0.5),
    ),
    (
        jnp.array(3),
        (jnp.array(3.0), jnp.array(5)),  # initial point on boundary
        -jnp.array(4.0),  # step outside allowed region
        None,
        jnp.array(0.0),
    ),
    (
        jnp.array(3),
        (-jnp.array(jnp.inf), jnp.array(5)),
        jnp.array(-1.0),
        None,
        jnp.array(1.0),
    ),
    (
        jnp.array(3),
        (jnp.array(-5), jnp.array(jnp.inf)),
        jnp.array(-4.0),
        None,
        jnp.array(1.0),
    ),
    (
        jnp.array(3),
        (jnp.array(-1), jnp.array(jnp.inf)),
        jnp.array(-8.0),
        None,
        jnp.array(0.5),
    ),
    (
        jnp.array(3),
        (-jnp.array(jnp.inf), jnp.array(5)),
        jnp.array(1.0),
        jnp.array(0.5),
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
        (-jnp.array([jnp.inf, jnp.inf]), jnp.array([3.0, 4.0])),
        jnp.array([1.0, 2.0]),
        None,
        jnp.array(1.0),
    ),
    (
        (jnp.array(1), jnp.array(2)),  # simply pytree structure
        ((-jnp.array(jnp.inf), -jnp.array(jnp.inf)), (jnp.array(3), jnp.array(4))),
        (jnp.array(1), jnp.array(2)),
        None,
        jnp.array(1.0),
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
)


barrier_values__y0_bounds_barrier_parameter_expected_result = (
    # y0, bounds, barrier_parameter, expected_result
    (
        jnp.ones((2,)),
        (0 * jnp.ones((2,)), 2 * jnp.ones((2,))),
        1.0,
        jnp.array(0.0),
    ),
    (
        0.5 * jnp.ones((2,)),
        (0 * jnp.ones((2,)), 2 * jnp.ones((2,))),
        1.0,
        jnp.array(0.57536414),
    ),
    (
        1.5 * jnp.ones((2,)),
        (0 * jnp.ones((2,)), 2 * jnp.ones((2,))),
        1.0,
        jnp.array(0.57536414),
    ),
)


## Smoke tests for all combinations of bound, equality and inequality constraints
# Tests that we hit all code paths correctly in interior point solvers that adjust these
# to the kinds of constraints present


def equality(y):
    x1, x2 = y
    return x1 - x2


def non_blocking_inequality(y):
    _, x2 = y
    return x2 + 1.2


def blocking_inequality(y):
    _, x2 = y
    return x2 - 1


non_blocking_bounds = (-2 * jnp.ones(2), 2 * jnp.ones(2))
blocking_bounds = (0.5 * jnp.ones(2), 2 * jnp.ones(2))


combinatorial_smoke__fn_y0_constraint_bounds_expected_result = (
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        None,
        None,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), None),
        None,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), non_blocking_inequality(y)),
        None,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (None, non_blocking_inequality(y)),
        None,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        None,
        non_blocking_bounds,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), None),
        non_blocking_bounds,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), non_blocking_inequality(y)),
        non_blocking_bounds,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (None, non_blocking_inequality(y)),
        non_blocking_bounds,
        jnp.zeros(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        None,
        blocking_bounds,
        0.5 * jnp.ones(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), None),
        blocking_bounds,
        0.5 * jnp.ones(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), non_blocking_inequality(y)),
        blocking_bounds,
        0.5 * jnp.ones(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (None, non_blocking_inequality(y)),
        blocking_bounds,
        0.5 * jnp.ones(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), blocking_inequality(y)),
        blocking_bounds,
        jnp.array([1.0, 1.0]),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (None, blocking_inequality(y)),
        blocking_bounds,
        jnp.array([0.5, 1.0]),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (equality(y), blocking_inequality(y)),
        non_blocking_bounds,
        jnp.ones(2),
    ),
    (
        _paraboloid,
        1.5 * jnp.ones(2),
        lambda y: (None, blocking_inequality(y)),
        non_blocking_bounds,
        jnp.array([0.0, 1.0]),
    ),
)
