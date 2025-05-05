from typing import Any, Generic

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
# other pots are burning, to use a german phrase. The inequality multipliers are also
# restricted with respect to the values they may take (currently strictly negative), so
# this would require a non-negative least squares solve for them, or some other kind of
# correction.
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
    least squares solve, again assuming optimality of the Lagrangian, which means that

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
    # TODO: this is currently a bit duplicate with code elsewhere! We initialise a slack
    # variable here, but this might not be the best idea - this should be done in one
    # place. And that place is probably not here.
    # If I'm not initialising the slack variables here, then I also do not need to have
    # this function be in the know about the truncation to the strict interior.
    # (So we can remove the hard-coded parameters below.)
    slack = inequality_residual
    lower = tree_full_like(slack, 0.0)
    upper = tree_full_like(slack, jnp.inf)
    slack = _interior_tree_clip(slack, lower, upper, 0.01, 0.01)
    inequality_multipliers = (-iterate.barrier / slack**ω).ω
    # jax.debug.print("slack: {}", slack)
    # jax.debug.print("inequality multipliers: {}", inequality_multipliers)

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
        return jnp.where(jnp.abs(x) > 1e3, x, 0.0)

    safe_equality_multipliers = jtu.tree_map(truncate, equality_multipliers)
    safe_inequality_multipliers = jtu.tree_map(truncate, inequality_multipliers)
    safe_duals = (safe_equality_multipliers, safe_inequality_multipliers)

    return Iterate(
        iterate.y_eval,
        iterate.slack,  # TODO: we're not currently updating the slack variables here!
        safe_duals,  # Only multipliers were updated
        iterate.bound_multipliers,
        iterate.barrier,
    )


# TODO: Find a better name for this system!
def _make_kkt_operator_xdycyd(iterate, f_info) -> tuple[lx.FunctionLinearOperator, Any]:
    """Construct the KKT operator for the primal-dual descent.

    This KKT system is equivalent to the `XDYcYd` system used in HiOp.

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

    slack = iterate.slack
    slack_bounds = (tree_full_like(slack, 0.0), tree_full_like(slack, jnp.inf))
    slack_barrier = LogarithmicBarrier(slack_bounds)

    # Set up the KKT operator
    equality_multipliers, inequality_multipliers = iterate.multipliers
    barrier_hessian, _ = slack_barrier.primal_dual_hessians(
        slack, (inequality_multipliers, inequality_multipliers)
    )

    hessian = f_info.hessian
    equality_jacobian, inequality_jacobian = f_info.constraint_jacobians

    def kkt_operator(inputs):
        y, slacks, (equality_multipliers, inequality_multipliers) = inputs
        r1 = (
            hessian.mv(y) ** ω
            + equality_jacobian.T.mv(equality_multipliers) ** ω
            + inequality_jacobian.T.mv(inequality_multipliers) ** ω
            + 1e-6 * tree_full_like(y, 1.0) ** ω
        ).ω
        r2 = (-(barrier_hessian.mv(slacks) ** ω) + inequality_multipliers**ω).ω
        r3 = equality_jacobian.mv(y)
        r4 = (inequality_jacobian.mv(y) ** ω + slacks**ω).ω
        return r1, r2, (r3, r4)

    # Set up the right-hand side of the KKT system
    equality_grad = equality_jacobian.T.mv(equality_multipliers)  # pyright: ignore
    inequality_grad = inequality_jacobian.T.mv(inequality_multipliers)  # pyright: ignore
    constraint_gradient = (equality_grad**ω + inequality_grad**ω).ω
    r1 = (f_info.grad**ω + constraint_gradient**ω).ω

    barrier_grad, _ = slack_barrier.grads(slack, iterate.barrier)  # One-sided for slack
    r2 = (inequality_multipliers**ω - barrier_grad**ω).ω

    equality_residual, inequality_residual = f_info.constraint_residual
    r3 = equality_residual
    r4 = (inequality_residual**ω - slack**ω).ω

    vector = (-((r1, r2, (r3, r4)) ** ω)).ω
    input_structure = jax.eval_shape(lambda: vector)

    return lx.FunctionLinearOperator(kkt_operator, input_structure), vector


def _postprocess_step_xdycyd(linear_solution, iterate):
    """Postprocess the step to ensure that it is feasible."""

    ## REFACTOR THIS: into a function that truncates the steps ---------------------
    # This function should also handle all the postprocessing
    # Challenge: this is also a property of the linear system that we use.
    # For example, if signs of elements are flipped to make the operator symmetric,
    # then this will vary depending on how we set up the linear system.
    # So it might be a good idea to have a class for the linear system, that has
    # attributes for making the operator, the vector, and for the postprocessing.
    # This class is then restricted to taking an iterate and a function info object,
    # but otherwise only specifies the signature of its methods.
    # The advantage is then that we can have a descent attribute that specifies
    # which system should be used, and otherwise the descent remains the same.
    # This would also help us when we might want to iteratively refine on a
    # different system, and it would be clear where things are tweaked.
    # The IPOPTLike descent would then always expect that a postprocessed iterate
    # consists of a feasible iterate.
    boundary_multipliers = iterate.bound_multipliers

    y_step, slack_steps, dual_steps = linear_solution

    slack_steps = (-(slack_steps**ω)).ω
    max_slack_step_size = feasible_step_length(
        iterate.slack,
        slack_steps,
        lower_bound=tree_full_like(iterate.slack, 0.0),
        upper_bound=tree_full_like(iterate.slack, jnp.inf),
        offset=iterate.barrier,
    )

    _, inequality_multipliers = iterate.multipliers
    _, inequality_multiplier_steps = dual_steps
    max_multiplier_step_size = feasible_step_length(
        inequality_multipliers,
        inequality_multiplier_steps,
        lower_bound=tree_full_like(inequality_multipliers, -jnp.inf),
        upper_bound=tree_full_like(inequality_multipliers, 0.0),
        offset=iterate.barrier,  # TODO: adaptive offset, I think that got lost (tau)
    )

    step_sizes = jnp.array([max_slack_step_size, max_multiplier_step_size])
    max_step_size = jnp.min(step_sizes)  # Monkey patch

    result = RESULTS.where(
        jnp.asarray(max_step_size) > 0.0,  # TODO: some cutoff value
        RESULTS.successful,  # use result from linear solve here
        RESULTS.feasibility_restoration_required,
    )

    iterate_step_ = Iterate(
        y_step,
        slack_steps,
        dual_steps,
        tree_full_like(boundary_multipliers, 0.0),  # TODO: not yet part of this descent
        tree_full_like(iterate.barrier, 0.0),
    )

    return (max_step_size * iterate_step_**ω).ω, result


def _leaf_from_struct(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jnp.zeros(x.shape, x.dtype)
    else:
        return x


class _XDYcYdDescentState(
    eqx.Module, Generic[Y, EqualityOut, InequalityOut], strict=True
):
    first_step: Bool[Array, ""]
    step: PyTree
    result: RESULTS
    linear_solver_state: PyTree  # TODO: how to type this?


# TODO: This is not strict anymore, since I'm experimenting with the correct method
# (@John this means we're not using a strict Equinox module)
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
        # We want to cache the state of the linear solver for use in corrective steps.
        # To do this, we first need to construct it - which requires reconstructing the
        # function information with dummy values, since f_info_struct contains
        # jax.ShapeDtypeStruct objects where we later need to have arrays.
        reconstructed_f_info = jtu.tree_map(_leaf_from_struct, f_info_struct)
        operator, _ = _make_kkt_operator_xdycyd(iterate, reconstructed_f_info)
        linear_solver_state = self.linear_solver.init(operator, {})

        return _XDYcYdDescentState(
            jnp.array(True),
            iterate,
            RESULTS.successful,
            linear_solver_state,
        )

    def query(  # pyright: ignore  (figuring out what types a descent should take)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _XDYcYdDescentState,
    ) -> _XDYcYdDescentState:
        # Special casing first steps: at the beginning of the optimisation, and after
        # a feasibility restoration.
        def keep_multipliers(iterate__f_info):
            iterate, _ = iterate__f_info
            return iterate

        args = (iterate, f_info)
        iterate = filter_cond(
            state.first_step, _initialise_multipliers, keep_multipliers, args
        )

        operator, vector = _make_kkt_operator_xdycyd(iterate, f_info)
        # TODO: regularisation / inertia correction comes in here
        out = lx.linear_solve(operator, vector, self.linear_solver)
        result = RESULTS.promote(out.result)

        # TODO: result: if max step size is 0, request feasibility restoration
        # now I need to decide which result should take precedence
        # step sizes of zero are not detected by the search, the search step size is
        # only a relative scaling of y_descent, so if y_descent is zero or near zero,
        # then the search will not detect this in our setup.
        # For unconstrained minimisation, this would not be a problem since it indicates
        # convergence and the solver terminates if it detects it. Here, I let the
        # function that handles postprocessing of the step request a feasibility
        # restoration if the step size is too small. We should think about where this
        # would be best placed.
        iterate_step, result = _postprocess_step_xdycyd(out.value, iterate)

        return _XDYcYdDescentState(
            jnp.array(False),
            iterate_step,
            result,
            out.state,
        )

    def correct(  # pyright: ignore  (figuring out what types a descent should take)
        self,
        iterate,
        f_info: FunctionInfo.EvalGradHessian,
        state: _XDYcYdDescentState,
    ) -> _XDYcYdDescentState:
        operator, vector = _make_kkt_operator_xdycyd(iterate, f_info)
        out = lx.linear_solve(
            operator, vector, self.linear_solver, state=state.linear_solver_state
        )
        result = RESULTS.promote(out.result)
        iterate_step = _postprocess_step_xdycyd(out.value, iterate)
        return _XDYcYdDescentState(
            jnp.array(False),
            iterate_step,
            result,
            out.state,
        )

    def step(self, step_size: Scalar, state: _XDYcYdDescentState) -> tuple[Y, RESULTS]:
        # TODO Note that I *am* currently scaling the dual variables for the bounds too
        # Fixing this would require unpacking the tuple and packing it up again
        # When very large values for the bounds are used, it does seem restrictive to
        # couple the boundary multipliers to the evolution of the primal and dual
        # variables - at least I have some empirical evidence that this can hinder
        # convergence. Might be useful to match IPOPT behaviour here.
        return (step_size * state.step**ω).ω, state.result
