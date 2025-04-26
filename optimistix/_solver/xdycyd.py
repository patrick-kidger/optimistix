from typing import Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import (
    EqualityOut,
    InequalityOut,
    Y,
)
from .._misc import (
    feasible_step_length,
    filter_cond,
    tree_full_like,
)
from .._search import (
    AbstractDescent,
    FunctionInfo,
)
from .._solution import RESULTS
from .barrier import LogarithmicBarrier
from .ipoptlike import (
    _interior_tree_clip,  # pyright: ignore (private function)
    Iterate,  # pyright: ignore
)


# TODO: After feasibility restoration (?) IPOPT apparently sets all multipliers to zero
# IIUC: https://coin-or.github.io/Ipopt/OPTIONS.html
# Should we do this here? Then we would need to set separate cutoff values for the
# initial and subsequent initialisations of the multipliers.
# They now also support an option to set the bound multipliers to mu/value, as we do
# below for the slack variables.
# ...and it turns out that they do their linear least squares solve for all multipliers,
# not just the equality multipliers. So this would require a change here. I'm tabling
# this for now though, since its not clear that it makes this much of a difference and
# other pots are burning, to use a german phrase.
def _initialise_multipliers(iterate__f_info):
    """Initialise the multipliers for the equality and inequality constraints. Both are
    initialised to the value they are expected to take at the optimum, with a safety
    factor for truncation if the computed values are unexpectedly large.

    The multipliers for the inequality constraints are initialised to the value they are
    expected to take at the optimum, which is

        inequality_multiplier = barrier_parameter / slack

    where the slack variables convert the inequality constraints to equality constraints
    such that we have

        g(y) - slack = 0.

    For an inequality constraint function `g`. The slack variables must always be
    strictly positive.

    We can then solve for the multipliers of the equality constraints with a linear
    least squares solve, again assuming optimality of the Lagragian, which means that

        grad Lagrangian = grad f + Jac h^T * l + Jac g^T * m = 0

    for an objective function `f`, equality constraints `h`, and inequality constraints
    `g`, with inequality constraint multipliers `m` computed as above. We then get

        Jac h^T * l = -grad f - Jac g^T * m

    which we can solve for the equality constraint multipliers `l`. We can do better
    than that though, since we know that the role of the Lagrangian multipliers is to
    counterbalance the gradient of the objective function at the optimum. This means
    that directions of the "constraint gradient" Jac h^T * l that are orthogonal to the
    gradient of the objective function can be discarded.
    To accomplish this, we introduce an auxiliary variable `w` that we require to be in
    the null space of the Jacobian of the equality constraints. We obtain the linear
    system

        [     I    Jac h^T ] [ w ] = [ -grad f - Jac g^T * m]
        [ Jac h^T    0     ] [ l ] = [          0           ]

    which we solve by least squares.

    Finally, strong linear dependence of the equality constraints can result in very
    large values for the multipliers computed in this way. To guard against this, we
    truncate all multipliers to a cutoff value, including the multipliers for the
    inequality constraints.

    Note: (TODO): IPOPT supports separate options for the `bound_frac` and `bound_push`
    parameters of the interior clipping function. Right now we do not support this and
    instead use a (hard-coded) value of 0.01 for both. The respective values are set as
    solver options, so patching them through to the descent would need to be figured
    out. Alternatively the value of the barrier parameter could be used, to make sure
    that we move the slack variables less as the overall solve progresses. However, this
    somewhat contradicts the purpose of the initialisation here - which should be
    robust, since it is done at the start and after every feasibility restoration step.
    """
    iterate, f_info = iterate__f_info
    (equality_multipliers, _) = iterate.multipliers
    _, inequality_residual = f_info.constraint_residual

    # TODO: special case for when we don't have any inequality constraints
    slack = inequality_residual
    lower = tree_full_like(slack, 0.0)
    upper = tree_full_like(slack, jnp.inf)
    slack = _interior_tree_clip(slack, lower, upper, 0.01, 0.01)
    inequality_multipliers = (iterate.barrier / slack**ω).ω

    equality_jacobian, inequality_jacobian = f_info.constraint_jacobians
    inequality_gradient = inequality_jacobian.T.mv(inequality_multipliers)
    gradient = (-(f_info.grad**ω) - inequality_gradient**ω).ω
    null = tree_full_like(equality_multipliers, 0.0)
    vector = (gradient, null)
    del equality_multipliers

    def make_operator(equality_jacobian, input_structure):
        jac = equality_jacobian  # for brevity

        def operator(inputs):
            orthogonal_component, parallel_component = inputs
            r1 = (orthogonal_component**ω + jac.T.mv(parallel_component) ** ω).ω
            r2 = jac.mv(orthogonal_component)
            return r1, r2

        return lx.FunctionLinearOperator(operator, input_structure)

    operator = make_operator(equality_jacobian, jax.eval_shape(lambda: vector))
    # TODO: we could allow a different choice of linear solver here
    out = lx.linear_solve(operator, vector, lx.SVD())
    _, equality_multipliers = out.value

    # TODO: hard-coded cutoff value for now! In IPOPT this is an option to the solver
    def truncate(x):
        return jnp.where(jnp.abs(x) < 1e3, x, 0.0)

    safe_equality_multipliers = jtu.tree_map(truncate, equality_multipliers)
    safe_inequality_multipliers = jtu.tree_map(truncate, inequality_multipliers)
    safe_duals = (safe_equality_multipliers, safe_inequality_multipliers)

    return Iterate(
        iterate.y_eval,
        iterate.slack,
        safe_duals,  # Only multipliers were updated
        iterate.bound_multipliers,
        iterate.barrier,
    )


class _XDYcYdDescentState(
    eqx.Module, Generic[Y, EqualityOut, InequalityOut], strict=True
):
    first_step: Bool[Array, ""]
    step: PyTree
    result: RESULTS


# TODO: This is not strict anymore, since I'm experimenting with the correct method
# TODO: This descent implements the XDYcYd descent, as implemented in HiOp. It currently
# requires that bounds be specified as inequality constraints.
# TODO: this thing NEEDS a new name, this name is maximally non-descriptive. (It stands
# for X - D - Yc - Yd, where X is the primal variable, D are the slack variables, Yc are
# the multipliers for the equality constraints and Yd are the multipliers for the
# inequality constraints.
class XDYcYdDescent(
    AbstractDescent[Y, FunctionInfo.EvalGradHessian, _XDYcYdDescentState],
):
    """TODO: add a description here."""

    linear_solver: lx.AbstractLinearSolver = lx.SVD()

    def init(  # pyright: ignore (figuring out what types a descent should take)
        self, iterate, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _XDYcYdDescentState:
        del f_info_struct
        # TODO: error management here
        return _XDYcYdDescentState(jnp.array(True), iterate, RESULTS.successful)

    def query(  # pyright: ignore  (figuring out what types a descent should take)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _XDYcYdDescentState,
    ) -> _XDYcYdDescentState:
        def keep_multipliers(iterate__f_info):
            iterate, _ = iterate__f_info
            return iterate

        args = (iterate, f_info)
        iterate = filter_cond(
            state.first_step, _initialise_multipliers, keep_multipliers, args
        )

        y = iterate.y_eval
        (equality_dual, inequality_dual) = iterate.multipliers
        boundary_multipliers = iterate.bound_multipliers
        barrier = iterate.barrier

        _, slack = f_info.constraint_residual  # pyright: ignore
        slack_bounds = (tree_full_like(slack, 0.0), tree_full_like(slack, jnp.inf))
        # TODO: values determining truncation to strict interior not used here yet
        slack = _interior_tree_clip(slack, *slack_bounds, 0.01, 0.01)
        slack_barrier = LogarithmicBarrier((slack_bounds))
        barrier_grad, _ = slack_barrier.grads(slack, barrier)
        barrier_hessian, _ = slack_barrier.primal_dual_hessians(
            slack, (inequality_dual, inequality_dual)
        )

        equality_residual, inequality_residual = f_info.constraint_residual  # pyright: ignore

        def make_kkt_operator(w, z, jacs):
            hj, gj = jacs

            # TODO: WIP: improvised inerta correction to see if this improves things.
            def kkt(inputs):
                y, slacks, (equality_duals, inequality_duals) = inputs
                y_pred = (
                    w.mv(y) ** ω
                    + hj.T.mv(equality_duals) ** ω
                    + gj.T.mv(inequality_duals) ** ω
                    + 0 * y**ω  # TODO: improvised inertia correction
                ).ω
                s_pred = (-(z.mv(slacks) ** ω) + inequality_duals**ω).ω
                hl_pred = (hj.mv(y) ** ω - 0 * equality_duals**ω).ω
                gl_pred = (gj.mv(y) ** ω + slacks**ω - 0 * inequality_duals**ω).ω
                return y_pred, s_pred, (hl_pred, gl_pred)

            return kkt

        kkt_operator_ = make_kkt_operator(
            f_info.hessian, barrier_hessian, f_info.constraint_jacobians
        )
        input_structure = jax.eval_shape(
            lambda: (
                y,
                slack,
                (equality_dual, inequality_dual),
            )
        )
        kkt_operator = lx.FunctionLinearOperator(kkt_operator_, input_structure)

        hj, gj = f_info.constraint_jacobians  # pyright: ignore
        eq_dual = equality_dual
        ineq_dual = inequality_dual
        constraint_gradient = (hj.T.mv(eq_dual) ** ω + gj.T.mv(ineq_dual) ** ω).ω  # pyright: ignore
        upper_rhs = (f_info.grad**ω + constraint_gradient**ω).ω

        vector = (
            upper_rhs,
            (inequality_dual**ω - barrier_grad**ω).ω,
            (equality_residual, (inequality_residual**ω - slack**ω).ω),
        )

        out = lx.linear_solve(kkt_operator, (-(vector**ω)).ω, self.linear_solver)

        # Output while I figure out the other steps
        y_step, slack_steps, dual_steps = out.value  # TODO: throw the slack steps away

        slack_steps = (-(slack_steps**ω)).ω
        slack_lower_bound, _ = slack_bounds
        max_slack_step_size = feasible_step_length(
            slack,
            slack_lower_bound,
            slack_steps,
            offset=barrier,
        )

        # TODO: constrain the step length for the inequality duals! Not sure whether
        # feasible_step_length does the right thing when the bound is an upper bound.
        max_step_size = max_slack_step_size

        # TODO: boundary multipliers currently do nothing / but not so fake anymore
        # Barrier parameter does not get updated, but it is something that is an iterate
        no_barrier_step = tree_full_like(barrier, 0.0)
        iterate_step = (y_step, dual_steps, boundary_multipliers, no_barrier_step)
        iterate_step = (max_step_size * iterate_step**ω).ω

        y_step, dual_steps, boundary_multipliers, no_barrier_step = iterate_step
        iterate_step = Iterate(
            y_step, slack_lower_bound, dual_steps, boundary_multipliers, no_barrier_step
        )

        return _XDYcYdDescentState(
            jnp.array(False),
            iterate_step,
            RESULTS.promote(out.result),
        )

    def correct(  # pyright: ignore  (figuring out what types a descent should take)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _XDYcYdDescentState,
    ) -> _XDYcYdDescentState:
        return self.query(iterate, f_info, state)  # TODO: cheaper implementation!

    def step(self, step_size: Scalar, state: _XDYcYdDescentState) -> tuple[Y, RESULTS]:
        # TODO Note that I *am* currently scaling the dual variables for the bounds too
        # Fixing this would require unpacking the tuple and packing it up again
        # When very large values for the bounds are used, it does seem restrictive to
        # couple the boundary multipliers to the evolution of the primal and dual
        # variables - at least I have some empirical evidence that this can hinder
        # convergence. Might be useful to match IPOPT behaviour here.
        return (step_size * state.step**ω).ω, state.result
