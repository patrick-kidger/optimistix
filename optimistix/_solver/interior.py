import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from .._custom_types import Y
from .._misc import feasible_step_length, tree_full_like, tree_min, tree_where
from .._search import AbstractDescent, FunctionInfo
from .._solution import RESULTS
from .barrier import LogarithmicBarrier


def _y_barrier__grad_operators(iterate, f_info):
    """Compute the gradients, primal dual Hessians, distance operators and the identity
    operators for the finite bound constraints.
    The distance operators are defined as diagonal operators that compute (y - lower)
    and (upper - y) for all elements with finite bounds. Finiteness operators indicate
    where the bounds are finite.
    """
    y = iterate.y_eval
    multipliers = iterate.bound_multipliers
    barrier_parameter = iterate.barrier

    barrier_function = LogarithmicBarrier(f_info.bounds)
    barrier_gradients = barrier_function.grads(y, barrier_parameter)
    barrier_hessians = barrier_function.primal_dual_hessians(y, multipliers)

    lower, upper = f_info.bounds
    finite_lower = jtu.tree_map(jnp.isfinite, lower)
    finite_upper = jtu.tree_map(jnp.isfinite, upper)

    distance_to_lower = lx.DiagonalLinearOperator(
        tree_where(finite_lower, (y**ω - lower**ω).ω, 0.0)
    )
    distance_to_upper = lx.DiagonalLinearOperator(
        tree_where(finite_upper, (upper**ω - y**ω).ω, 0.0)
    )
    distance_operators = (distance_to_lower, distance_to_upper)

    finite_lower = lx.DiagonalLinearOperator(finite_lower)
    finite_upper = lx.DiagonalLinearOperator(finite_upper)
    finite_operators = (finite_lower, finite_upper)

    return barrier_gradients, (barrier_hessians, distance_operators, finite_operators)


def _slack_barrier_derivatives(iterate):
    slack = iterate.slack
    _, inequality_multipliers = iterate.multipliers
    barrier_parameter = iterate.barrier

    # Hacky: dummy bound + multiplier b/c barrier currently expects two-sided bounds
    slack_bounds = (tree_full_like(slack, 0.0), tree_full_like(slack, jnp.inf))
    slack_barrier = LogarithmicBarrier(slack_bounds)

    # One-sided for slack variables, since upper bound is infinity
    gradient, _ = slack_barrier.grads(slack, barrier_parameter)
    multipliers = inequality_multipliers, tree_full_like(inequality_multipliers, 0.0)
    hessian, _ = slack_barrier.primal_dual_hessians(slack, multipliers)
    return gradient, hessian


def _lagrangian_gradient(iterate, f_info):
    """Compute the gradient of the Lagrangian, which is the sum of the objective and
    any constraints, weighted by the multipliers. This is the upper block row of the
    right-hand-side in uncondensed KKT systems.

    For a Lagrangian of the form

    L = f + h^T * λh + (g - s)^T * λg - (y - lb)^T * zL - (ub - y)^T * zU

    with equality constraints h and inequality constraints g, the gradient is given by

    ∇L = ∇f + ∇h^T * λh + ∇g^T * λg - zL + zU  (w.r.t. y)

    This is what we compute here. Note that we subtract the bound constraint terms to
    conform to the sign convention for barrier terms, while the other two constraint
    terms are added.
    """
    grad = f_info.grad

    if f_info.constraint_jacobians is not None:
        equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
        equality_multipliers, inequality_multipliers = iterate.multipliers

        if equality_jacobian is not None:
            equality_grad = equality_jacobian.T.mv(equality_multipliers)
            grad = (grad**ω + equality_grad**ω).ω
        if inequality_jacobian is not None:
            inequality_grad = inequality_jacobian.T.mv(inequality_multipliers)
            grad = (grad**ω + inequality_grad**ω).ω

    if f_info.bounds is not None:
        lower, upper = f_info.bounds
        finite_lower = jtu.tree_map(jnp.isfinite, lower)
        finite_upper = jtu.tree_map(jnp.isfinite, upper)

        lower_multiplier, upper_multiplier = iterate.bound_multipliers
        lower_multiplier = tree_where(finite_lower, lower_multiplier, 0.0)  # Safeguard
        upper_multiplier = tree_where(finite_upper, upper_multiplier, 0.0)

        grad = (grad**ω - lower_multiplier**ω + upper_multiplier**ω).ω
    return grad


def _iterate_bounds(iterate, f_info):
    """Defines an upper and lower bound for the given iterate structure. These bounds
    specify that:

    - the optimisation variable `y` stays within its bounds
    - slack variables remain strictly positive
    - multipliers for inequality constraints remain strictly negative
    - multipliers for bound constraints remain strictly positive
    """
    iterate_lower = tree_full_like(iterate, -jnp.inf)
    iterate_upper = tree_full_like(iterate, jnp.inf)

    if f_info.bounds is not None:
        lower, upper = f_info.bounds
        iterate_lower = eqx.tree_at(lambda i: i.y_eval, iterate_lower, lower)
        iterate_upper = eqx.tree_at(lambda i: i.y_eval, iterate_upper, upper)
    if iterate.slack is not None:
        positive = tree_full_like(iterate.slack, 0.0)
        iterate_lower = eqx.tree_at(lambda i: i.slack, iterate_lower, positive)
    if iterate.multipliers is not None:
        equality_multipliers, inequality_multipliers = iterate.multipliers
        upper = (
            tree_full_like(equality_multipliers, jnp.inf),
            tree_full_like(inequality_multipliers, 0.0),
        )
        iterate_upper = eqx.tree_at(lambda i: i.multipliers, iterate_upper, upper)
    if iterate.bound_multipliers is not None:
        lower = tree_full_like(iterate.bound_multipliers, 0.0)
        iterate_lower = eqx.tree_at(lambda i: i.bound_multipliers, iterate_lower, lower)

    return iterate_lower, iterate_upper


def _maybe_truncate(iterate, iterate_step, f_info):
    """Truncate the computed step to respect the bound constraints on `y`, as well as
    on the slack variables and any inequality multipliers (including the multipliers
    for the bound constraints).
    The `variable_step_sizes` argument specifies for which variables the step size may
    differ from the minimum of the all step sizes. Supported options are "none", which
    leads to all variables being scaled using the minimum step size, and
    "bound-multipliers", which scales the bound multipliers with the minimum step size
    that allows them to remain strictly positive.
    """
    # TODO: this function could implement the minimum offset rule, where the minimum
    # offset from the boundary is updated based on the barrier parameter. In IPOPT, this
    # keeps the minimum offset from dropping below a value of 0.01.
    # It computes:
    # offset = min(minimum_offset, 1 - barrier_parameter)
    offset = 0.01

    # If bounds are infinite, then we set the associated multipliers and distances to
    # zero. However, the linear solve may still return tiny steps for these, due to
    # numerical errata. Here we set multipliers for infinite bounds to zero.
    if f_info.bounds is not None:
        lower, upper = f_info.bounds
        finite_lower = jtu.tree_map(jnp.isfinite, lower)
        finite_upper = jtu.tree_map(jnp.isfinite, upper)
        lm_step, um_step = iterate_step.bound_multipliers
        safe_lm_step = tree_where(finite_lower, lm_step, 0.0)
        safe_um_step = tree_where(finite_upper, um_step, 0.0)
        safe_multiplier_steps = (safe_lm_step, safe_um_step)
        iterate_step = eqx.tree_at(
            lambda i: i.bound_multipliers, iterate_step, safe_multiplier_steps
        )

    iterate_bounds = _iterate_bounds(iterate, f_info)
    max_step_sizes = feasible_step_length(
        iterate, iterate_step, *iterate_bounds, offset=offset, reduce=False
    )

    # TODO: check if all of the maximum step sizes are zero and return a boolean if that
    # is the case. This would allow us to do something about it, like requesting a
    # feasibility restoration.

    # TODO: this function could also implement support for different step sizes for the
    # different categories of variables. (IPOPT allows bound multipliers to vary
    # independently.)
    return (tree_min(max_step_sizes) * iterate_step**ω).ω


class _AbstractKKTSystem(eqx.Module):
    """Implements the appropriate linear system and any associated methods. This
    includes methods to construct the operator and right hand side, to reconstruct the
    steps in condensed variables if applicable, to apply iterative refinement and to
    truncate the step taken to its maximum feasible length.

    Why do we need so many flavors of this? With three categories of constraints -
    bound constraints, inequality constraints, and equality constraints - there are
    eight different problem types we're interested in solving here. (With and without
    each type of constraint.) Couldn't we go with some type of unifying operation here
    and only deal with one type of system? There are certainly things that could be
    done! For example, bound constraints could be expressed as general inequality
    constraints, and then we'd have one fewer thing to deal with.

    However, each type of constraint has a specific structure that we can leverage to
    solve larger systems more effectively.
    For example, the Jacobian of bound constraints is the identity, and they require no
    slack variables, since the distance to the bound is trivial to compute. This allows
    us to easily condense the system - which in this case means that we can recover the
    steps in the multipliers for the bound constraints from the steps in the primal
    variable `y`, and hence can get away with solving a smaller linear system.
    Inequality constraints, on the other hand, do generally require the introduction of
    slack variables to specify a distance from the constraint bound for multivariate
    constraints. Introducing trivial slack variables for the equality constraints would
    blow up our linear system for no good reason! So we buy efficiency with some degree
    of verbosity. (Hopefully, we're also buying ourselves some extra clarity.)
    (That said, if we can kill this and replace it with something simpler I'm very happy
    to do so! Suggestions welcome.)

    ...and why do we always include the full system? We could solve a reduced system,
    and map back to the full variable space (all relevant primal and dual variables),
    and then call it a day. However, there is a cheap method to improve our solutions
    that generally works better on the full system, especially in the presence of ill-
    conditioning. This is called iterative refinement, and it works by minimising the
    residual of the linear solve.

    Put more concretely, if we have an imprecise linear solution x, and Ax - b = r != 0,
    we can compute a corrective factor d by solving the linear system Ad = -r, since
    A(x + d) - b = 0  => Ad = -(Ax - b) = -r. Then we set x = x + d and repeat, either
    until the residual drops below some specified tolerance or some number of repeats.
    Since we have already factorised A, we do not need to do this again! If we've solved
    a reduced system A', we first need to map back to the original system A, compute a
    residual in that system, map it back to the reduced space and solve for a corrective
    factor in the reduced space. (And then rinse and repeat.)

    This is particularly helpful when a reduced system was solved, especially if some of
    the optimisation variables or matrix elements that were condensed together take very
    large or very small values. Ill-conditioning of this type is introduced by the
    barrier Hessians, elements of which may take very large values near the boundary.
    Another frequent source of ill-conditioning are linearly dependent rows in the
    constraint Jacobian. Having the full system accessible means that we can apply
    iterative refinement if we so choose.

    The full KKT system is given by

    [HessL   0   Jac^T h   Jac^T g  -I    I] [ Δy]     [        grad L         ]
    [  0    -BS     0         I      0    0] [-Δs]     [ -grad barrier(s) - λg ]
    [Jac h   0      0         0      0    0] [Δλh] = - [         h(y)          ]
    [Jac g   I      0         0      0    0] [Δλg]     [       g(y) - s        ]
    [  ZL    0      0         0     DL    0] [ΔzL]     [      DL zL - μ        ]
    [ -ZU    0      0         0      0   DU] [ΔzU]     [      DU zU - μ        ]

    where HessL is the Hessian of the Lagrangian, Jac h and Jac g are the Jacobians of
    the equality constraints and inequality constraints, respectively, and BS is the
    primal-dual barrier Hessian for the slack variables s. ZL and ZU are diagonal
    matrices containing the Lagrange multipliers for the finite bound constraints, and
    DL and DU are the diagonal matrices specifiying the (positive) distance to the lower
    and upper bounds, with DL = diag(y - lb) and DU = diag(ub - y).

    From this base system, irrelevant block rows are left out to avoid bloating the
    linear systems with large numbers of zero rows.
    """

    @abc.abstractmethod
    def _make_full_system(
        self, iterate, f_info
    ) -> tuple[lx.AbstractLinearOperator, PyTree[Any]]:
        """Create the "full" system - without any condensed variables - and its right-
        hand-side. This system is used during iterative refinement.
        This method must also return a linear operator and a vector whose structure must
        be equal to the input structure of the operator.
        """

    # TODO: I don't love that lx.Solutions are processed in here... but required if we
    # want to apply iterative refinement.
    @abc.abstractmethod
    def refine_and_truncate(
        self, iterate, f_info, out: lx.Solution
    ) -> tuple[Any, RESULTS]:  # TODO: this should return an iterate and result
        """Postprocess the result of the linear solve. In this method, the step is
        truncated to its maximum feasible value, and iterative refinement may be
        applied. If the system was solved in a reduced space, then the solution is
        mapped back to the full space. Elements of the iterate that are not updated in
        the linear solve (e.g. barrier parameters) get a zero step length.

        TODO: if the feasible step length is zero, should we change the result? If yes,
        then to what?
        """

    @abc.abstractmethod
    def make_operator_vector(
        self, iterate, f_info
    ) -> tuple[lx.AbstractLinearOperator, PyTree[Any]]:
        """Create the operator and vector used when solving the linear system. (This is
        the operator that will be factorised.) If the linear system is condensed in one
        or several or its variables, then this operator should be the condensed system.

        Returns a tuple with the linear operator in the first argument, and the
        right-hand-side in the second argument. The vector must correspond to the input
        structure of the operator, but may otherwise take a variety of shapes.
        """


class _UnconstrainedKKTSystem(_AbstractKKTSystem):
    """Implements a Newton step in an unconstrained system without any bounds. This
    should be needed rarely in practice, but represents an edge case we do support.
    """

    def _make_full_system(
        self,
        iterate,
        f_info,
    ) -> tuple[lx.AbstractLinearOperator, PyTree[Any]]:
        del iterate
        return f_info.hessian, (-(f_info.grad**ω)).ω

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement - or skip for this case?
        del f_info
        y_step = out.value
        iterate_step = tree_full_like(iterate, 0.0)  # Default to no steps
        iterate_step = eqx.tree_at(lambda i: i.y_eval, iterate_step, y_step)
        result = RESULTS.promote(out.result)
        return iterate_step, result


class _BoundedUnconstrainedKKTSystem(_AbstractKKTSystem):
    """Implements a KKT system with bounds, but without any other constraints.
    The full system is:

    [HessL   0      -I    I] [ Δy]     [        grad L         ]
    [  ZL    0      DL    0] [ΔzL]     [      DL zL - μ        ]
    [ -ZU    0       0   DU] [ΔzU]     [      DU zU - μ        ]

    When condensing the bounds, we obtain the reduced system:

    # TODO
    """

    condense_bounds: bool

    def _make_full_system(self, iterate, f_info):
        _, y_barrier_operators = _y_barrier__grad_operators(iterate, f_info)
        hessians, distances, finiteness = y_barrier_operators
        lower_bound_hessian, upper_bound_hessian = hessians
        distance_to_lower, distance_to_upper = distances
        finite_lower, finite_upper = finiteness

        def operator(inputs):
            y_step, bound_multiplier_step = inputs
            lower_multiplier_step, upper_multiplier_step = bound_multiplier_step

            r1 = (
                f_info.hessian.mv(y_step) ** ω
                - finite_lower.mv(lower_multiplier_step) ** ω
                + finite_upper.mv(upper_multiplier_step) ** ω
            ).ω
            r2 = (
                lower_bound_hessian.mv(y_step) ** ω
                + distance_to_lower.mv(lower_multiplier_step) ** ω
            ).ω
            r3 = (
                -(upper_bound_hessian.mv(y_step) ** ω)
                + distance_to_upper.mv(upper_multiplier_step) ** ω
            ).ω
            return r1, (r2, r3)

        # Compute the right-hand side with current values
        r1 = _lagrangian_gradient(iterate, f_info)

        lower_multiplier, upper_multiplier = iterate.bound_multipliers
        r2 = (distance_to_lower.mv(lower_multiplier) - iterate.barrier**ω).ω
        r3 = (distance_to_upper.mv(upper_multiplier) - iterate.barrier**ω).ω

        vector = (r1, (r2, r3))
        vector = (-(vector**ω)).ω
        input_structure = jax.eval_shape(lambda: vector)

        return lx.FunctionLinearOperator(operator, input_structure), vector

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement

        y_step, m_steps = out.value
        step = tree_full_like(iterate, 0.0)
        step = eqx.tree_at(lambda i: i.y_eval, step, y_step)
        step = eqx.tree_at(lambda i: i.bound_multipliers, step, m_steps)
        step = _maybe_truncate(iterate, step, f_info)

        # TODO: different result if max feasible step length is zero?
        # If we were to allow for different step lengths in the different variables,
        # should we even curtail if one of them can't move? I don't think so?
        result = RESULTS.promote(out.result)
        return step, result


class _BoundedInequalityConstrainedKKTSystem(_AbstractKKTSystem):
    """Implements a KKT system with bounds and inequality constraints, but without
    equality constraints.

    [HessL   0   Jac^T g  -I    I] [ Δy]     [        grad L         ]
    [  0    -BS     I      0    0] [-Δs]     [ -grad barrier(s) - λg ]
    [Jac g   I      0      0    0] [Δλg]     [       g(y) - s        ]
    [  ZL    0      0     DL    0] [ΔzL]     [      DL zL - μ        ]
    [ -ZU    0      0      0   DU] [ΔzU]     [      DU zU - μ        ]
    """

    condense_bounds: bool

    def _make_full_system(self, iterate, f_info):
        _, y_barrier_operators = _y_barrier__grad_operators(iterate, f_info)
        hessians, distances, finiteness = y_barrier_operators
        lower_bound_hessian, upper_bound_hessian = hessians
        distance_to_lower, distance_to_upper = distances
        finite_lower, finite_upper = finiteness

        slack_barrier_grad, slack_barrier_hessian = _slack_barrier_derivatives(iterate)
        _, inequality_jacobian = f_info.constraint_jacobians

        def operator(inputs):
            (
                y_step,
                slack_step,
                inequality_multiplier_step,
                bound_multiplier_step,
            ) = inputs
            lower_multiplier_step, upper_multiplier_step = bound_multiplier_step

            r1 = (
                f_info.hessian.mv(y_step) ** ω
                + inequality_jacobian.T.mv(inequality_multiplier_step) ** ω
                - finite_lower.mv(lower_multiplier_step) ** ω
                + finite_upper.mv(upper_multiplier_step) ** ω
            ).ω
            r2 = (
                -(slack_barrier_hessian.mv(slack_step) ** ω)
                + inequality_multiplier_step**ω
            ).ω
            r3 = (inequality_jacobian.mv(y_step) ** ω + slack_step**ω).ω
            r4 = (
                lower_bound_hessian.mv(y_step) ** ω
                + distance_to_lower.mv(lower_multiplier_step) ** ω
            ).ω
            r5 = (
                -(upper_bound_hessian.mv(y_step) ** ω)
                + distance_to_upper.mv(upper_multiplier_step) ** ω
            ).ω
            return r1, r2, r3, (r4, r5)

        # Compute the right-hand side with current values
        r1 = _lagrangian_gradient(iterate, f_info)

        _, inequality_multipliers = iterate.multipliers
        r2 = (inequality_multipliers**ω - slack_barrier_grad**ω).ω

        _, inequality_residual = f_info.constraint_residual
        slack = iterate.slack
        r3 = (inequality_residual**ω - slack**ω).ω

        lower_multiplier, upper_multiplier = iterate.bound_multipliers
        r4 = (distance_to_lower.mv(lower_multiplier) ** ω - iterate.barrier**ω).ω
        r5 = (distance_to_upper.mv(upper_multiplier) ** ω - iterate.barrier**ω).ω

        vector = (r1, r2, r3, (r4, r5))
        vector = (-(vector**ω)).ω
        input_structure = jax.eval_shape(lambda: vector)

        return lx.FunctionLinearOperator(operator, input_structure), vector

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement

        y_step, slack_step, multiplier_step, bound_multiplier_step = out.value
        slack_step = (-(slack_step**ω)).ω
        multiplier_step = (None, multiplier_step)  # No equality constraints

        step = tree_full_like(iterate, 0.0)
        step = eqx.tree_at(lambda i: i.y_eval, step, y_step)
        step = eqx.tree_at(lambda i: i.slack, step, slack_step)
        step = eqx.tree_at(lambda i: i.multipliers, step, multiplier_step)
        step = eqx.tree_at(lambda i: i.bound_multipliers, step, bound_multiplier_step)
        step = _maybe_truncate(iterate, step, f_info)

        # TODO: different result if max feasible step length is zero?
        # If we were to allow for different step lengths in the different variables,
        # should we even curtail if one of them can't move? I don't think so?
        result = RESULTS.promote(out.result)
        return step, result


class _BoundedEqualityInequalityConstrainedKKTSystem(_AbstractKKTSystem):
    """Implements a KKT system for the case where we have bound, inequality and equality
    constraints.

    [HessL   0   Jac^T h   Jac^T g  -I    I] [ Δy]     [        grad L         ]
    [  0    -BS     0         I      0    0] [-Δs]     [ -grad barrier(s) - λg ]
    [Jac h   0      0         0      0    0] [Δλh] = - [         h(y)          ]
    [Jac g   I      0         0      0    0] [Δλg]     [       g(y) - s        ]
    [  ZL    0      0         0     DL    0] [ΔzL]     [      DL zL - μ        ]
    [ -ZU    0      0         0      0   DU] [ΔzU]     [      DU zU - μ        ]
    """

    condense_bounds: bool

    def _make_full_system(self, iterate, f_info):
        _, y_barrier_operators = _y_barrier__grad_operators(iterate, f_info)
        hessians, distances, finiteness = y_barrier_operators
        lower_bound_hessian, upper_bound_hessian = hessians
        distance_to_lower, distance_to_upper = distances
        finite_lower, finite_upper = finiteness

        slack_barrier_grad, slack_barrier_hessian = _slack_barrier_derivatives(iterate)
        equality_jacobian, inequality_jacobian = f_info.constraint_jacobians

        def operator(inputs):
            y_step, slack_step, multiplier_step, bound_multiplier_step = inputs
            lower_multiplier_step, upper_multiplier_step = bound_multiplier_step
            equality_multiplier_step, inequality_multiplier_step = multiplier_step

            r1 = (
                f_info.hessian.mv(y_step) ** ω
                + equality_jacobian.T.mv(equality_multiplier_step) ** ω
                + inequality_jacobian.T.mv(inequality_multiplier_step) ** ω
                - finite_lower.mv(lower_multiplier_step) ** ω
                + finite_upper.mv(upper_multiplier_step) ** ω
            ).ω
            r2 = (
                -(slack_barrier_hessian.mv(slack_step) ** ω)
                + inequality_multiplier_step**ω
            ).ω
            r3 = equality_jacobian.mv(y_step)
            r4 = (inequality_jacobian.mv(y_step) ** ω + slack_step**ω).ω
            r5 = (
                lower_bound_hessian.mv(y_step) ** ω
                + distance_to_lower.mv(lower_multiplier_step) ** ω
            ).ω
            r6 = (
                -(upper_bound_hessian.mv(y_step) ** ω)
                + distance_to_upper.mv(upper_multiplier_step) ** ω
            ).ω
            return r1, r2, (r3, r4), (r5, r6)

        # Compute the right-hand side with current values
        r1 = _lagrangian_gradient(iterate, f_info)

        _, inequality_multipliers = iterate.multipliers
        r2 = (inequality_multipliers**ω - slack_barrier_grad**ω).ω

        equality_residual, inequality_residual = f_info.constraint_residual
        slack = iterate.slack
        r3 = equality_residual
        r4 = (inequality_residual**ω - slack**ω).ω

        lower_multiplier, upper_multiplier = iterate.bound_multipliers
        r5 = (distance_to_lower.mv(lower_multiplier) ** ω - iterate.barrier**ω).ω
        r6 = (distance_to_upper.mv(upper_multiplier) ** ω - iterate.barrier**ω).ω

        vector = (r1, r2, (r3, r4), (r5, r6))
        vector = (-(vector**ω)).ω
        input_structure = jax.eval_shape(lambda: vector)

        return lx.FunctionLinearOperator(operator, input_structure), vector

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement

        y_steps, slack_steps, multiplier_steps, bound_multiplier_steps = out.value
        slack_steps = (-(slack_steps**ω)).ω

        step = tree_full_like(iterate, 0.0)
        step = eqx.tree_at(lambda i: i.y_eval, step, y_steps)
        step = eqx.tree_at(lambda i: i.slack, step, slack_steps)
        step = eqx.tree_at(lambda i: i.multipliers, step, multiplier_steps)
        step = eqx.tree_at(lambda i: i.bound_multipliers, step, bound_multiplier_steps)
        step = _maybe_truncate(iterate, step, f_info)

        # TODO: different result if max feasible step length is zero?
        # If we were to allow for different step lengths in the different variables,
        # should we even curtail if one of them can't move? I don't think so?
        result = RESULTS.promote(out.result)
        return step, result


class _BoundedEqualityConstrainedKKTSystem(_AbstractKKTSystem):
    """Implement a KKT system when bounds and equality constraints are present, but no
    inequality constraints (and hence no slack variables).

    [HessL   Jac^T h   -I    I] [ Δy]     [    grad L   ]
    [Jac h      0       0    0] [Δλh] = - [     h(y)    ]
    [  ZL       0      DL    0] [ΔzL]     [  DL zL - μ  ]
    [ -ZU       0       0   DU] [ΔzU]     [  DU zU - μ  ]
    """

    condense_bounds: bool

    def _make_full_system(self, iterate, f_info):
        _, y_barrier_operators = _y_barrier__grad_operators(iterate, f_info)
        hessians, distances, finiteness = y_barrier_operators
        lower_bound_hessian, upper_bound_hessian = hessians
        distance_to_lower, distance_to_upper = distances
        finite_lower, finite_upper = finiteness

        equality_jacobian, _ = f_info.constraint_jacobians

        def operator(inputs):
            y_step, equality_multiplier_step, bound_multiplier_step = inputs
            lower_multiplier_step, upper_multiplier_step = bound_multiplier_step

            r1 = (
                f_info.hessian.mv(y_step) ** ω
                + equality_jacobian.T.mv(equality_multiplier_step) ** ω
                - finite_lower.mv(lower_multiplier_step) ** ω
                + finite_upper.mv(upper_multiplier_step) ** ω
            ).ω
            r2 = equality_jacobian.mv(y_step)
            r3 = (
                lower_bound_hessian.mv(y_step) ** ω
                + distance_to_lower.mv(lower_multiplier_step) ** ω
            ).ω
            r4 = (
                -(upper_bound_hessian.mv(y_step) ** ω)
                + distance_to_upper.mv(upper_multiplier_step) ** ω
            ).ω
            return r1, r2, (r3, r4)

        # Compute the right-hand side with current values
        r1 = _lagrangian_gradient(iterate, f_info)

        equality_residual, _ = f_info.constraint_residual
        r2 = equality_residual

        lower_multiplier, upper_multiplier = iterate.bound_multipliers
        r3 = (distance_to_lower.mv(lower_multiplier) ** ω - iterate.barrier**ω).ω
        r4 = (distance_to_upper.mv(upper_multiplier) ** ω - iterate.barrier**ω).ω

        vector = (r1, r2, (r3, r4))
        vector = (-(vector**ω)).ω
        input_structure = jax.eval_shape(lambda: vector)

        return lx.FunctionLinearOperator(operator, input_structure), vector

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement

        y_step, multiplier_step, bound_multiplier_step = out.value
        multiplier_step = (multiplier_step, None)  # No inequality constraints

        step = tree_full_like(iterate, 0.0)
        step = eqx.tree_at(lambda i: i.y_eval, step, y_step)
        step = eqx.tree_at(lambda i: i.multipliers, step, multiplier_step)
        step = eqx.tree_at(lambda i: i.bound_multipliers, step, bound_multiplier_step)
        step = _maybe_truncate(iterate, step, f_info)

        # TODO: different result if max feasible step length is zero?
        # If we were to allow for different step lengths in the different variables,
        # should we even curtail if one of them can't move? I don't think so?
        result = RESULTS.promote(out.result)
        return step, result


class _InequalityConstrainedKKTSystem(_AbstractKKTSystem):
    """Implement a KKT system when only inequality constraints are present."""

    def _make_full_system(self, iterate, f_info):
        slack_barrier_grad, slack_barrier_hessian = _slack_barrier_derivatives(iterate)
        _, inequality_jacobian = f_info.constraint_jacobians

        def operator(inputs):
            y_step, slack_step, inequality_multiplier_step = inputs

            r1 = (
                f_info.hessian.mv(y_step) ** ω
                + inequality_jacobian.T.mv(inequality_multiplier_step) ** ω
            ).ω
            r2 = (
                -(slack_barrier_hessian.mv(slack_step) ** ω)  # TODO (-) sign needed?
                + inequality_multiplier_step**ω
            ).ω
            r3 = (inequality_jacobian.mv(y_step) ** ω + slack_step**ω).ω
            return r1, r2, r3

        # Compute the right-hand side with current values
        r1 = _lagrangian_gradient(iterate, f_info)

        _, inequality_multipliers = iterate.multipliers
        r2 = (inequality_multipliers**ω - slack_barrier_grad**ω).ω

        _, inequality_residual = f_info.constraint_residual
        slack = iterate.slack
        r3 = (inequality_residual**ω - slack**ω).ω

        vector = (r1, r2, r3)
        vector = (-(vector**ω)).ω
        input_structure = jax.eval_shape(lambda: vector)

        return lx.FunctionLinearOperator(operator, input_structure), vector

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement

        y_step, slack_step, multiplier_step = out.value
        slack_step = (-(slack_step**ω)).ω
        multiplier_step = (None, multiplier_step)  # No equality constraints

        step = tree_full_like(iterate, 0.0)
        step = eqx.tree_at(lambda i: i.y_eval, step, y_step)
        step = eqx.tree_at(lambda i: i.slack, step, slack_step)
        step = eqx.tree_at(lambda i: i.multipliers, step, multiplier_step)
        step = _maybe_truncate(iterate, step, f_info)

        # TODO: different result if max feasible step length is zero?
        # If we were to allow for different step lengths in the different variables,
        # should we even curtail if one of them can't move? I don't think so?
        result = RESULTS.promote(out.result)
        return step, result


class _EqualityConstrainedKKTSystem(_AbstractKKTSystem):
    """Implement a KKT system when only equality constraints are present."""

    def _make_full_system(self, iterate, f_info):
        equality_jacobian, _ = f_info.constraint_jacobians

        def operator(inputs):
            y_step, equality_multiplier_step = inputs

            r1 = (
                f_info.hessian.mv(y_step) ** ω
                + equality_jacobian.T.mv(equality_multiplier_step) ** ω
            ).ω
            r2 = equality_jacobian.mv(y_step)
            return r1, r2

        # Compute the right-hand side with current values
        r1 = _lagrangian_gradient(iterate, f_info)

        equality_residual, _ = f_info.constraint_residual
        r2 = equality_residual

        vector = (r1, r2)
        vector = (-(vector**ω)).ω
        input_structure = jax.eval_shape(lambda: vector)

        return lx.FunctionLinearOperator(operator, input_structure), vector

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement

        y_step, multiplier_step = out.value
        multiplier_step = (multiplier_step, None)  # No inequality constraints

        step = tree_full_like(iterate, 0.0)
        step = eqx.tree_at(lambda i: i.y_eval, step, y_step)
        step = eqx.tree_at(lambda i: i.multipliers, step, multiplier_step)
        step = _maybe_truncate(iterate, step, f_info)

        # TODO: different result if max feasible step length is zero?
        # If we were to allow for different step lengths in the different variables,
        # should we even curtail if one of them can't move? I don't think so?
        result = RESULTS.promote(out.result)
        return step, result


# TODO: perhaps place citation somewhere more prominent
class _EqualityInequalityConstrainedKKTSystem(_AbstractKKTSystem):
    """Implement a KKT system when equality and inequality constraints are present, but
    no (separate) bound constraints are specified. (We don't check if the specified
    inequality constraints contain any reformulated bound constraints.)

    The system used here is equivalent to the `XDYcYd` system used in HiOp.

    ??? cite "References"

        @TECHREPORT{hiop_techrep,
            title={{HiOp} -- {U}ser {G}uide},
            author={Petra, Cosmin G. and Chiang, NaiYuan and Jingyi Wang},
            year={2018},
            institution = {Center for Applied Scientific Computing,
                           Lawrence Livermore National Laboratory},
            number = {LLNL-SM-743591}
        }
    """

    def _make_full_system(self, iterate, f_info):
        slack_barrier_grad, slack_barrier_hessian = _slack_barrier_derivatives(iterate)
        equality_jacobian, inequality_jacobian = f_info.constraint_jacobians

        def operator(inputs):
            y_step, slack_step, multiplier_step = inputs
            equality_multiplier_step, inequality_multiplier_step = multiplier_step

            r1 = (
                f_info.hessian.mv(y_step) ** ω
                + equality_jacobian.T.mv(equality_multiplier_step) ** ω
                + inequality_jacobian.T.mv(inequality_multiplier_step) ** ω
            ).ω
            r2 = (
                -(slack_barrier_hessian.mv(slack_step) ** ω)  # TODO (-) sign needed?
                + inequality_multiplier_step**ω
            ).ω
            r3 = equality_jacobian.mv(y_step)
            r4 = (inequality_jacobian.mv(y_step) ** ω + slack_step**ω).ω
            return r1, r2, (r3, r4)

        # Compute the right-hand side with current values
        r1 = _lagrangian_gradient(iterate, f_info)

        _, inequality_multipliers = iterate.multipliers
        r2 = (inequality_multipliers**ω - slack_barrier_grad**ω).ω

        equality_residual, inequality_residual = f_info.constraint_residual
        slack = iterate.slack
        r3 = equality_residual
        r4 = (inequality_residual**ω - slack**ω).ω

        vector = (r1, r2, (r3, r4))
        vector = (-(vector**ω)).ω
        input_structure = jax.eval_shape(lambda: vector)

        return lx.FunctionLinearOperator(operator, input_structure), vector

    def make_operator_vector(self, iterate, f_info):
        return self._make_full_system(iterate, f_info)

    def refine_and_truncate(self, iterate, f_info, out):
        # TODO: iterative refinement

        y_steps, slack_steps, multiplier_steps = out.value
        slack_steps = (-(slack_steps**ω)).ω

        step = tree_full_like(iterate, 0.0)
        step = eqx.tree_at(lambda i: i.y_eval, step, y_steps)
        step = eqx.tree_at(lambda i: i.slack, step, slack_steps)
        step = eqx.tree_at(lambda i: i.multipliers, step, multiplier_steps)
        step = _maybe_truncate(iterate, step, f_info)

        # TODO: different result if max feasible step length is zero?
        # If we were to allow for different step lengths in the different variables,
        # should we even curtail if one of them can't move? I don't think so?
        result = RESULTS.promote(out.result)
        return step, result


def _get_system(iterate, f_info, condense_bounds: bool):
    """Choose the linear system to use based on the available function information. We
    have eight options, for three categories of constraints (bounds, inequalities, and
    equalities), which all may be present or absent.
    In this function, we check if they are present and return the appropriate linear
    system to use when computing descent steps.
    """
    assert isinstance(f_info, FunctionInfo.EvalGradHessian)  # TODO error message

    if f_info.bounds is None:
        if f_info.constraint_jacobians is None:
            return _UnconstrainedKKTSystem()
        else:
            equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
            if equality_jacobian is None:
                assert inequality_jacobian is not None
                return _InequalityConstrainedKKTSystem()
            elif inequality_jacobian is None:
                assert equality_jacobian is not None
                return _EqualityConstrainedKKTSystem()
            else:
                return _EqualityInequalityConstrainedKKTSystem()
    else:
        if f_info.constraint_jacobians is None:
            return _BoundedUnconstrainedKKTSystem(condense_bounds)
        else:
            equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
            if equality_jacobian is None:
                assert inequality_jacobian is not None
                return _BoundedInequalityConstrainedKKTSystem(condense_bounds)
            elif inequality_jacobian is None:
                assert equality_jacobian is not None
                return _BoundedEqualityConstrainedKKTSystem(condense_bounds)
            else:
                return _BoundedEqualityInequalityConstrainedKKTSystem(condense_bounds)


def _leaf_from_struct(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jnp.zeros(x.shape, x.dtype)  # We need arrays, not ShapeDtypeStructs
    else:
        return x


class _InteriorDescentState(eqx.Module):
    step: PyTree[Any]  # While we sort out what the iterates look like
    result: RESULTS
    linear_solver_state: PyTree[Any]


class NewInteriorDescent(
    AbstractDescent[Y, FunctionInfo.EvalGradHessian, _InteriorDescentState],
):
    """A primal-dual descent through the interior of the feasible set. Primal-dual means
    that we update the primal variables `y` and any slack variables alongside the dual
    variables, or Lagrange multipliers. Taking a step through the interior of the
    feasible region means that, given the available function information and constraint
    residuals, this descent computes a step that does not result in infeasibility with
    respect to the bound constraints, slack variables and Lagrange multipliers for the
    bound and inequality constraints.
    The step computation relies on a local linearization of the equality and inequality
    constraints, so these may be violated by the computed step.

    There are many ways to compute interior steps, and a combination of cases we might
    wish to solve. This includes any of the eight possible combinations of constraints -
    bound, inequality and equality constraints may all be present or not - as well as
    specific choices for setting up the linear system, such as solving it in a reduced
    space, or applying iterative refinement.
    This descent will pick the linear system based on the available function information
    and additional options specifying how the system may be reduced.

    All systems use barrier terms for bound constraints and slack variables, where
    applicable.
    """

    linear_solver: lx.AbstractLinearSolver = lx.SVD()
    condense_bounds: bool = False  # TODO toggle to True once this is supported

    def init(  # pyright: ignore (figuring out what iterate should be typed as)
        self, iterate, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _InteriorDescentState:
        # We want to cache the state of the linear solver for use in corrective steps.
        # To initialise the linear solver state, we need to reconstruct the function
        # information with dummy values.
        reconstructed_f_info = jtu.tree_map(_leaf_from_struct, f_info_struct)
        system = _get_system(iterate, reconstructed_f_info, self.condense_bounds)
        jax.debug.print("system: {}", system)

        operator, _ = system.make_operator_vector(iterate, reconstructed_f_info)
        linear_solver_state = self.linear_solver.init(operator, {})
        return _InteriorDescentState(iterate, RESULTS.successful, linear_solver_state)

    def query(  # pyright: ignore
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _InteriorDescentState,
    ) -> _InteriorDescentState:
        del state
        system = _get_system(iterate, f_info, self.condense_bounds)

        # TODO: If needed, initialise the multipliers
        operator, vector = system.make_operator_vector(iterate, f_info)
        out = lx.linear_solve(operator, vector, self.linear_solver)

        step, result = system.refine_and_truncate(iterate, f_info, out)
        return _InteriorDescentState(step, result, out.state)

    def correct(  # pyright: ignore
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _InteriorDescentState,
    ) -> _InteriorDescentState:
        return state  # TODO: Not tested yet
        system = _get_system(iterate, f_info, self.condense_bounds)
        assert system is not None

        operator, vector = system.make_operator_vector(iterate, f_info)
        out = lx.linear_solve(
            operator, vector, self.linear_solver, state=state.linear_solver_state
        )

        step, result = system.refine_and_truncate(iterate, f_info, out)
        return _InteriorDescentState(step, result, state.linear_solver_state)

    def step(
        self, step_size: Scalar, state: _InteriorDescentState
    ) -> tuple[Y, RESULTS]:
        # TODO Note that I *am* currently scaling the dual variables for the bounds too
        # Fixing this would require unpacking the tuple and packing it up again
        # When very large values for the bounds are used, it does seem restrictive to
        # couple the boundary multipliers to the evolution of the primal and dual
        # variables - at least I have some empirical evidence that this can hinder
        # convergence. Might be useful to match IPOPT behaviour here.
        return (step_size * state.step**ω).ω, state.result


NewInteriorDescent.__init__.__doc__ = """**Arguments**: 

- `linear_solver`: the linear solver to use.
- `condense_bounds`: whether to condense the block rows pertaining to the bound
    constraints. If True, then the steps in the bound multipliers are recovered from
    the steps in the primal variable `y`. If False, then they are computed directly
    during the linear solve. If the problem does not have bound constraints, then this
    option is ignored.
"""
