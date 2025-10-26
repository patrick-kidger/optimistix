import functools as ft
from typing import Generic

import equinox as eqx
import equinox.internal as eqxi
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, Float

from .._custom_types import Y
from .._misc import filter_cond, tree_clip, tree_dot


def _boundary_intercepts(y: Y, lower: Y, upper: Y, grad: Y):
    """Compute the intercepts of the gradient with the hyperplanes defined by the
    bounds on y, as fractions of the full gradient step.

    Intercepts are computed as

        (y - bound) / gradient > 0 and gradient != 0, else jnp.inf

    under the assumption that y is always inside the feasible region defined by the
    bounds. This works because

     -  if the gradient < 0, taking a step will increase y, so there could be an
        intercept. (y - upper bound) is a negative value, resulting in a positive
        quotient.
     -  if the gradient > 0, we're taking steps that decrease y. (y - lower bound) is
        a positive value as well.

    In all other cases we would either move away from the bounds or not move at all.

    Reference: Nocedal and Wright, Numerical Optimization, (2006). p. 485 ff.
    """

    def leaf_intercepts(y, bound, gradient):
        intercepts = (y - bound) / gradient
        intercepts = jnp.where(intercepts > 0, intercepts, jnp.inf)
        intercepts = jnp.where(gradient != 0.0, jnp.asarray(intercepts), jnp.inf)
        return intercepts

    lower_intercepts = jtu.tree_map(leaf_intercepts, y, lower, grad)
    upper_intercepts = jtu.tree_map(leaf_intercepts, y, upper, grad)

    intercepts = jtu.tree_map(
        lambda x, y: jnp.minimum(x, y), lower_intercepts, upper_intercepts
    )
    intercepts, _ = jfu.ravel_pytree(intercepts)
    return intercepts


class _CauchyState(eqx.Module, Generic[Y]):
    intercepts: Float[Array, "..."]
    t_prev: Float[Array, ""]
    point: Y
    terminate: Bool[Array, ""]


def _find_cauchy_point(
    y: Y,
    lower: Y,
    upper: Y,
    grad: Y,
    hessian_operator: lx.AbstractLinearOperator,
    intercepts: Float[Array, "..."],
) -> Y:
    grad_dot = ft.partial(tree_dot, grad)

    def cond_fun(state):
        return jnp.invert(state.terminate)

    # For each intercept, we update the trial step (`y` plus some step of length t along
    # the piecewise linear path) and check if this is a minimiser. We determine if we
    # should continue searching or stop, and replace the intercept we just checked with
    # jnp.inf before checking another - this way we can always determine the nearest
    # intercepting point by just taking the minimum of all values of t.
    # In the following, `grad` is the gradient of the target function, while `quad_grad`
    # is the gradient of the quadratic approximation to the target function.
    def body_fun(state):
        min_index = jnp.argmin(state.intercepts)
        t_next = state.intercepts[min_index]

        next_point = (y**ω - t_next * grad**ω).ω
        next_point = tree_clip(next_point, lower, upper)
        point_diff = (next_point**ω - state.point**ω).ω

        quad_grad = grad_dot(point_diff) + tree_dot(
            state.point, hessian_operator.mv(point_diff)
        )
        quad_hess = tree_dot(point_diff, hessian_operator.mv(point_diff))

        t_opt = -quad_grad / quad_hess
        stop_at_prev = quad_grad > 0  # continue only if still descending
        t_opt = jnp.array(jnp.where(stop_at_prev, state.t_prev, t_opt))
        skip_to_next = t_opt > t_next  # don't overshoot!
        t_opt = jnp.array(jnp.where(skip_to_next, t_next, t_opt))

        is_cauchy = stop_at_prev | jnp.invert(skip_to_next)

        point = (y**ω - t_opt * grad**ω).ω
        point = tree_clip(point, lower, upper)

        # Overwrite current value and any duplicates with inf
        intercepts = state.intercepts
        intercepts = jnp.asarray(jnp.where(intercepts == t_next, jnp.inf, intercepts))
        more_to_try = jnp.any(jnp.isfinite(intercepts))

        terminate = is_cauchy | jnp.invert(more_to_try)

        return _CauchyState(
            intercepts=intercepts,
            t_prev=t_next,
            point=point,
            terminate=terminate,
        )

    init_state = _CauchyState(intercepts, jnp.array(0.0), y, jnp.array(False))
    final_state = eqxi.while_loop(cond_fun, body_fun, init_state, kind="lax")

    return final_state.point


def cauchy_point(
    y: Y, lower: Y, upper: Y, grad: Y, hessian_operator: lx.AbstractLinearOperator
) -> Y:
    """Compute the (generalised) Cauchy point, the first local minimiser along the
    piecewise linear path obtained by projecting the gradient onto the hypercube
    defined by the bound constraints.

    To find this point, we minimise

        q(Δy(t)) = grad^T Δy(t) + 1/2 Δy(t)^T Hess Δy(t)

    in t, which may be understood as a fractional step length.

    Once we have located the generalised Cauchy point, we use it as our educated
    guess at the set of active bound constraints at the solution, by simply assuming
    that all elements on the surface of the hypercube (at their respective bounds)
    are blocked by these bounds, treating all elements in the interior as
    unconstrained. We may then optimise the free variables in the unconstrained
    subspace.

    This function is used by [`optimistix.BFGSB`][] and [`optimistix.LBFGSB`][].
    The usage of second order information from the Hessian approximations, in
    combination with the subsequent subspace optimisation, guarantee favourable
    convergence properties for bounded minimisation problems.

    This function assumes that lower < upper holds for all bound elements, but this
    property is not checked.

    In our naming of this function, we follow the naming convention used e.g. in
    Conn, Gould and Toint's "Trust Region Methods". Other authors, including Nocedal
    + Wright, call this method "Projected Gradient", a name we deliberately avoid
    since it is nowadays mostly used to describe first-order gradient based methods
    with after-the-step projections onto feasible sets with more general simple
    geometries, such as balls or spheres.
    """

    intercepts = _boundary_intercepts(y, lower, upper, grad)
    has_intercepts = jnp.any(jnp.isfinite(intercepts))

    # TODO(jhaffner): minimise the number of arguments passed to these two branches
    def project(args):
        y, grad, hessian_operator, intercepts = args
        return _find_cauchy_point(y, lower, upper, grad, hessian_operator, intercepts)

    # Rather than solving for the optimal step length using the quadratic
    # approximation, we just take a gradient step if the full gradient step does not
    # intersect the surface of the hypercube. The reason for this is that the Cauchy
    # point is used to determine the set of active constraints, rather than as an
    # iterate itself. We are therefore interested in its location relative to the
    # bounds, which just requires taking a step.
    # (We should move since we might currently be at the boundary.)
    def just_move(args):
        y, grad, _, _ = args
        return (y**ω - grad**ω).ω

    args = (y, grad, hessian_operator, intercepts)
    cauchy = filter_cond(has_intercepts, project, just_move, args)
    return cauchy
