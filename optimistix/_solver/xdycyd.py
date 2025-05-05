import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω

from .._misc import (
    tree_full_like,
)
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
