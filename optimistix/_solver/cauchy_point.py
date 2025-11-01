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
from .._misc import feasible_step_length, filter_cond, tree_clip, tree_dot, tree_min


def _boundary_intercepts(y: Y, lower: Y, upper: Y, grad: Y):
    """Compute the intercepts of the gradient with the bounds on y, as fractions of
    the full gradient step per element of `y`. These are computed as

        (y - bound) / gradient > 0 and gradient != 0, else jnp.inf

    under the assumption that y is always inside the feasible region defined by the
    bounds. This then works because

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
        intercepts = jnp.where(intercepts >= 0, intercepts, jnp.inf)
        intercepts = jnp.where(gradient != 0.0, jnp.asarray(intercepts), jnp.inf)
        return intercepts

    lower_intercepts = jtu.tree_map(leaf_intercepts, y, lower, grad)
    upper_intercepts = jtu.tree_map(leaf_intercepts, y, upper, grad)

    all_intercepts = jtu.tree_map(
        lambda x, y: jnp.minimum(x, y), lower_intercepts, upper_intercepts
    )
    all_intercepts, _ = jfu.ravel_pytree(all_intercepts)
    return all_intercepts


def _next_intercept(intercepts, previous_intercept):
    """The next intercept is the first finite intercept that is larger than the
    previous one. We're inverting that and taking the argmin to get the first
    element for which this condition evaluates to True.
    """
    candidates = jnp.invert(intercepts > previous_intercept)
    candidates = jnp.where(
        jnp.isfinite(intercepts), candidates, jnp.ones_like(candidates)
    )
    return intercepts[jnp.argmin(candidates)]


def _point(y, grad, intercept, lower, upper):
    return tree_clip((y**ω - intercept * grad**ω).ω, lower, upper)


class _CauchyState(eqx.Module, Generic[Y]):
    intercepts: Float[Array, "..."]
    previous_intercept: Float[Array, ""]
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
    """To identify the minimiser `t_opt` of the univariate quadratic function q(y(t)),
    we use Newton's method on each line segment between two element-wise boundary
    intercepts. (The linear part is linear in the displacement Δt along the line
    segment, the quadratic part is, well, quadratic in Δt.)
    """

    def cond_fun(state):
        return jnp.invert(state.terminate)

    def body_fun(state):
        next_intercept = _next_intercept(state.intercepts, state.previous_intercept)
        next_point = _point(y, grad, next_intercept, lower, upper)
        point_diff = (next_point**ω - y**ω).ω
        linear = tree_dot(grad, point_diff) + tree_dot(
            state.point, hessian_operator.mv(point_diff)
        )
        quadratic = tree_dot(point_diff, hessian_operator.mv(point_diff))
        optimal_intercept = -linear / quadratic

        stop_at_previous = linear > 0  # Enforce descent condition along segment
        skip_to_next = optimal_intercept > next_intercept
        optimal_intercept = jnp.array(
            jnp.where(stop_at_previous, state.previous_intercept, optimal_intercept)
        )
        optimal_intercept = jnp.array(
            jnp.where(skip_to_next, next_intercept, optimal_intercept)
        )
        is_cauchy = stop_at_previous | jnp.invert(skip_to_next)

        point = _point(y, grad, optimal_intercept, lower, upper)
        more_to_try = jnp.any(state.intercepts > next_intercept)

        return _CauchyState(
            intercepts=intercepts,
            previous_intercept=next_intercept,
            point=point,
            terminate=is_cauchy | jnp.invert(more_to_try),
        )

    init_state = _CauchyState(intercepts, jnp.array(0.0), y, jnp.array(False))
    final_state = eqxi.while_loop(cond_fun, body_fun, init_state, kind="lax")

    return final_state.point


def cauchy_point(
    y: Y, lower: Y, upper: Y, grad: Y, hessian_operator: lx.AbstractLinearOperator
) -> Y:
    """Compute the (generalised) Cauchy point, the first local minimiser along the
    piecewise linear path obtained by projecting the gradient onto the box defined
    by the bound constraints.

    To find this point, we minimise

        q(y(t)) = f(y) + grad^T Δy(t) + 1/2 Δy(t)^T Hess Δy(t)

    in y(t), where t quantifies displacement along the piecewise linear path.

    Given some box, we can picture the directions + points computed: the gradient
    step, the path defined by the projected gradient, and the Cauchy point somewhere
    along this path:

                   Δ
                  /
                 /
                /
               /
              /
             /
     -------/==*==>     º   Initial point / current `y`
     |     /      |     /   gradient step
     |    /       |     ==> projected gradient step
     |   /        |     *   Cauchy point
     |  º         |     _|  box outlines
     |____________|

    Once we have located the generalised Cauchy point, we use it as our educated
    guess at the set of active bound constraints at the solution, by simply assuming
    that all elements on the surface of the hypercube (at their respective bounds)
    are blocked by these bounds, treating all elements in the interior as
    unconstrained. We may then optimise the free variables in the unconstrained
    subspace.

    This function is used by [`optimistix.BFGSB`][] and [`optimistix.LBFGSB`][]. It
    assumes that lower < upper holds for all bound elements, but this property is not
    checked.

    ??? cite "References"

        This implementation follows

        ```bibtex
        @article{byrd1995bounded,
            author = {
                Byrd, Richard H. and Lu, Peihuang and Nocedal, Jorge and Zhu, Ciyou
            },
            title = {A Limited Memory Algorithm for Bound Constrained Optimization},
            journal = {SIAM Journal on Scientific Computing},
            volume = {16},
            number = {5},
            pages = {1190-1208},
            year = {1995},
            doi = {10.1137/0916069},
        }
        ```
    """
    intercepts = _boundary_intercepts(y, lower, upper, grad)
    # Iteratively search for the Cauchy point if there are finite intercept values > 0
    has_intercepts = jnp.any(jnp.logical_and(jnp.isfinite(intercepts), intercepts > 0))

    def project(_y):
        return _find_cauchy_point(_y, lower, upper, grad, hessian_operator, intercepts)

    # There are three possible cases in which we do not project the gradient onto the
    # surface of the box. In each case, there is no finite, nonzero intercept.
    # 1) We are in the interior and a full gradient step would not reach the boundary.
    # 2) We are on the boundary and a gradient step would take us away from it.
    # 3) We are on the boundary and any gradient step would lead outside of it.
    #
    # In the third case, there is at least one zero-valued intercept, whereas in the
    # first two there are only infinite-valued intercepts. Since we are usually only
    # interested in the location of the Cauchy point relative to the bounds of the
    # feasible region, taking a gradient step suffices in cases 1) and 2). In the third
    # case we guard against leaving the feasible set.
    def just_move(_y):
        gradient_step = (-(grad**ω)).ω
        step_size = feasible_step_length(_y, gradient_step, lower, upper)
        return (_y**ω + tree_min(step_size) * gradient_step**ω).ω

    cauchy = filter_cond(has_intercepts, project, just_move, y)
    return cauchy
