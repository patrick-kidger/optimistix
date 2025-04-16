from collections.abc import Callable
from typing import Generic

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from .._custom_types import Aux, Out, Y
from .._misc import (
    feasible_step_length,
    filter_cond,
    max_norm,
    tree_clip,
    tree_dot,
    tree_full_like,
    tree_where,
)
from .._quadratic_solve import AbstractQuadraticSolver
from .._search import AbstractDescent, FunctionInfo
from .._solution import RESULTS
from .learning_rate import LearningRate


def _boundary_intercepts(y: Y, f_info: FunctionInfo.EvalGradHessian):  # pyright: ignore
    """Compute the intercepts of the gradient with the hyperplanes defined by the
    bounds on y.

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

    lower, upper = f_info.bounds  # pyright: ignore
    lower_intercepts = jtu.tree_map(leaf_intercepts, y, lower, f_info.grad)
    upper_intercepts = jtu.tree_map(leaf_intercepts, y, upper, f_info.grad)

    intercepts = jtu.tree_map(
        lambda x, y: jnp.minimum(x, y), lower_intercepts, upper_intercepts
    )
    intercepts = jnp.array(jtu.tree_leaves(intercepts)).flatten()
    return intercepts


class _CauchyState(eqx.Module, Generic[Y]):
    intercepts: Float[Array, "..."]
    t_prev: Float[Array, ""]
    point: Y
    terminate: Bool[Array, ""]


def _find_cauchy_point(
    y: Y, f_info: FunctionInfo.EvalGradHessian, intercepts: Float[Array, "..."]
) -> Y:
    lower, upper = f_info.bounds  # pyright: ignore
    grad_dot = f_info.compute_grad_dot
    hess_mv = f_info.hessian.mv

    def cond_fun(state):
        return jnp.invert(state.terminate)

    def body_fun(state):
        min_index = jnp.argmin(state.intercepts)
        t_next = state.intercepts[min_index]

        next_point = (y**ω - t_next * f_info.grad**ω).ω
        next_point = tree_clip(next_point, lower, upper)
        point_diff = (next_point**ω - state.point**ω).ω

        # TODO(jhaffner): This equation should be documented here. And grad, hess are
        # not (!) the best names here.
        grad = grad_dot(point_diff) + tree_dot(state.point, hess_mv(point_diff))
        hess = tree_dot(point_diff, hess_mv(point_diff))

        t_opt = -grad / hess
        stop_at_prev = grad > 0  # continue only if descending
        t_opt = jnp.array(jnp.where(stop_at_prev, state.t_prev, t_opt))
        skip_to_next = t_opt > t_next  # don't overshoot!
        t_opt = jnp.array(jnp.where(skip_to_next, t_next, t_opt))

        is_cauchy = stop_at_prev | jnp.invert(skip_to_next)

        point = (y**ω - t_opt * f_info.grad**ω).ω
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


def cauchy_point(y: Y, f_info: FunctionInfo.EvalGradHessian) -> Y:
    """Compute the Cauchy point, which can be used to make an educated guess of the set
    of bound constraints active at the solution. (This set may be empty.)

    To compute the Cauchy point, we first compute the intercepts of the gradient with
    the faces of the hypercube defined by the bounds. We then search for the first point
    that minimises the quadratic approximation of the objective along this piecewise
    linear path.

    That is, we minimise q (y(t)) = grad^T y(t) + 1/2 y(t)^T Hess y(t) in t, while
    updating y(t) in a step-wise manner to account for each boundary intercept. We then
    do a Newton solve for t between successive intercepting points, until we're either
    no longer descending along the projected path, or have run out of intercepts.

    Note that this function assumes that lower < upper holds for all bound elements.
    """

    if f_info.bounds is None:
        raise ValueError("A Cauchy point can only be computed if bounds are provided.")
    else:
        if isinstance(f_info, FunctionInfo.EvalGradHessian):
            intercepts = _boundary_intercepts(y, f_info)
            project = jnp.any(jnp.isfinite(intercepts))

            def projected(args):
                y, f_info, intercepts = args
                return _find_cauchy_point(y, f_info, intercepts)

            # Rather than solving for the optimal step length using the quadratic
            # approximation, we just take a gradient step here. The reason for this is
            # that the Cauchy point is usually used to determine the set of active
            # constraints, rather than as an iterate itself. We are therefore mostly
            # interested in its location relative to the bounds, which just requires
            # taking a step.
            # (We'd also have to special-case zero values of the second order term.)
            def in_bounds(args):
                y, f_info, _ = args
                return (y**ω - f_info.grad**ω).ω

            cauchy = filter_cond(project, projected, in_bounds, (y, f_info, intercepts))
            return cauchy

        else:
            raise NotImplementedError(
                "Cauchy point computation is currently only implemented for solvers "
                "that provide gradient and (approximate) Hessian information. "
                "(Note that inverted Hessians cannot yet be treated.)"
            )


# TODO: do we treat nonfinite bounds appropriately?
def cauchy_newton_step(y, f_info, linear_solver):
    cauchy = cauchy_point(y, f_info)
    lower, upper = f_info.bounds  # pyright: ignore (handled by cauchy_point)

    active_lower = jtu.tree_map(lambda a, b: a <= b, cauchy, lower)
    active_upper = jtu.tree_map(lambda a, b: a >= b, cauchy, upper)
    active_bound_values = tree_where(active_lower, lower, tree_full_like(lower, 0.0))
    active_bound_values = tree_where(active_upper, upper, active_bound_values)
    active = jtu.tree_map(lambda a, b: a | b, active_lower, active_upper)

    bound_constraint = lx.DiagonalLinearOperator(active)
    dual_constraint = lx.DiagonalLinearOperator(jtu.tree_map(jnp.invert, active))

    def make_kkt_function(hessian, bound_constraint, dual_constraint):
        def kkt_function(inputs):
            y, dual = inputs
            y_pred = (hessian.mv(y) ** ω - bound_constraint.transpose().mv(dual) ** ω).ω
            dual_pred = (bound_constraint.mv(y) ** ω + dual_constraint.mv(dual) ** ω).ω
            return y_pred, dual_pred

        return kkt_function

    kkt_function = make_kkt_function(f_info.hessian, bound_constraint, dual_constraint)
    in_struct = jax.eval_shape(lambda: (y, y))
    kkt_operator = lx.FunctionLinearOperator(kkt_function, in_struct)

    # Exact dual values: grad(f) = lambda * grad(c), where c is the constraint function.
    # Since we only have bound constraints, grad(c) is one where the constraint is
    # active and zero where it is not.
    dual = tree_where(active, f_info.grad, 0.0)
    upper_rhs = (bound_constraint.mv(dual) ** ω - f_info.grad**ω).ω
    lower_rhs = (active_bound_values**ω - bound_constraint.mv(y) ** ω).ω
    vector = upper_rhs, lower_rhs

    out = lx.linear_solve(kkt_operator, vector, linear_solver)
    cauchy_newton, _ = out.value
    result = RESULTS.promote(out.result)

    max_to_lower = feasible_step_length(y, lower, cauchy_newton)
    max_to_upper = feasible_step_length(y, upper, cauchy_newton)
    max_step_size = jnp.min(jnp.array([max_to_lower, max_to_upper, 1.0]))
    cauchy_newton = (max_step_size * cauchy_newton**ω).ω

    return cauchy_newton, result


class _CauchyNewtonDescentState(eqx.Module, Generic[Y], strict=True):
    cauchy_newton: Y
    result: RESULTS


class CauchyNewtonDescent(
    AbstractDescent[
        Y,
        FunctionInfo.EvalGradHessian,
        _CauchyNewtonDescentState,
    ],
    strict=True,
):
    """Computes the Cauchy point to make an educated guess of the set of active bound
    constraints. Then, a Newton step is computed using the KKT system, where the active
    bound constraints are enforced as equalities.

    This means that we are taking a traditional Newton step - towards the bottom of a
    quadratic bowl - in the unconstrained elements of `y`, and a projected gradient step
    in the constrained elements of `y`.
    """

    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)

    def init(  # pyright: ignore
        self, y: Y, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _CauchyNewtonDescentState:
        del f_info_struct
        return _CauchyNewtonDescentState(cauchy_newton=y, result=RESULTS.successful)

    def query(  # pyright: ignore
        self,
        y: Y,
        f_info: FunctionInfo.EvalGradHessian,
        state: _CauchyNewtonDescentState,
    ) -> _CauchyNewtonDescentState:
        del state
        cauchy_newton, result = cauchy_newton_step(y, f_info, self.linear_solver)
        return _CauchyNewtonDescentState(cauchy_newton=cauchy_newton, result=result)

    def step(
        self, step_size: Scalar, state: _CauchyNewtonDescentState
    ) -> tuple[Y, RESULTS]:
        # TODO(jhaffner): double check signs!
        return (step_size * state.cauchy_newton**ω).ω, state.result


class CauchyNewton(AbstractQuadraticSolver[Y, Out, Aux], strict=True):
    """A simple quadratic solver that projects the gradient onto the feasible set, which
    is defined by a hypercube, to find the first minimiser along the resulting piecewise
    linear path. It will then constrain all elements of `y` that are at their bounds,
    while leaving the other elements unconstrained.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    search: LearningRate[Y]
    descent: CauchyNewtonDescent[Y]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
    ):
        """**Arguments:**

        - `rtol`: Relative tolerance for terminating the solve.
        - `atol`: Absolute tolerance for terminating the solve.
        """
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.search = LearningRate(1.0)
        self.descent = CauchyNewtonDescent(lx.SVD())  # TODO(jhaffner): update solver
