import abc
from typing import Union

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import ScalarLike

from .._custom_types import Constraint, EqualityOut, InequalityOut, Y
from .._minimise import AbstractMinimiser, minimise
from .._misc import scalarlike_asarray, tree_clip, two_norm
from .._solution import RESULTS


# TODO: Small fix, but once we have everything else we need to consider where these
# things should show up in the documentation, and what the most intuitive order of
# presentation in the sidebar is. Currently we mix constrained and unconstrained
# minimisers, and since some minimisers will now respect constraints to some extend via
# boundary maps, this should be reflected in the documentation structure.
# TODO: I think we will need to pay attention to potentially infinite recursions in the
# computation graph, by requiring that a solver that solves a nonlinear problem defined
# by a boundary map does not itself have a boundary map, or if it does that this latter
# map be trivial / consist of an analytic solution? This sounds like it could lead to a
# compilation time footgun.
class AbstractBoundaryMap(eqx.Module, strict=True):
    """Abstract base class for boundary maps.

    These can be projections onto a feasible set with a trivial form, such as a box or
    a ball. They can also define more general projections onto a manifold or level set
    defined by a constraint function.
    Depending on the complexity of the feasible set, the computation required can be
    very simple and cheap - such as clipping `y` to bounds - or require a (non-)linear
    solve.
    """

    # TODO(jhaffner): ConstraintOut is now part of the documentation, I don't think it
    # should be, since this is a custom type that is internal to optimistix. This
    # requires the definition of a TypeAlias, I think. Do that down the line once we
    # have definitely settled on what the constraint function should return, and if
    # there is any advantage to not restricting this to (a tuple of) arrays.
    @abc.abstractmethod
    def __call__(
        self,
        y: Y,
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
    ) -> tuple[Y, RESULTS]:
        """Maps a point outside of the boundary of the feasible set back onto the
        feasible set.

        **Arguments**:

        - `y`: The point to be projected.
        - `constraint`: The constraint function defining the feasible set, or None if
            the feasible set is defined only by the bounds.
        - `bounds`: The bounds defining the feasible set, or None if the feasible set is
            defined only be the constraint function.

        **Returns**:

        - `y_projected`: The projected point.
        - `result`: The result of the projection, an instance of [optimistix.RESULTS][].
        """


class BoxProjection(AbstractBoundaryMap, strict=True):
    r"""Project any point onto a box defined by upper and lower bounds on the variables.
    Lower or upper bounds may be infinite, in which case the respective elements of `y`
    get returned unchanged.

    If the upper and lower bounds are both finite, then this is equivalent to the
    solution of

    $\min\limits_{p} ||y-p||_2^2 \quad s.t. \quad l_b \le p \le u_b$

    where `l_b` and `u_b` are the lower and upper bounds, respectively.
    Comparable to `optax.projections.projection_box`.

    Since a box defines a convex set, this projection is safe to use with a backtracking
    line search such as [optimistix.BacktrackingArmijo][] with `step_init <= 1.0`, or
    with [optimistix.LearningRate][] with `learning_rate <= 1.0`.
    """

    # TODO(jhaffner): include an example for the usage of this boundary map, or a
    # general one, in the documentation.

    def __call__(
        self,
        y: Y,
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
    ) -> tuple[Y, RESULTS]:
        del constraint
        if bounds is None:
            raise ValueError("Box projection requires bounds to be defined.")
        else:
            lower, upper = bounds
            return tree_clip(y, lower, upper), RESULTS.successful


# TODO adapt to equality + inequality constraints
def _make_scaled_deviation(y):
    scaling = jtu.tree_map(lambda x: 1 / jnp.array(x), y)
    scaling = jtu.tree_map(lambda x: jnp.clip(x, max=1.0), scaling)
    scaling_operator = lx.DiagonalLinearOperator(scaling)

    def squared_scaled_deviation(_y):
        ydiff = (_y**ω - y**ω).ω
        return two_norm(scaling_operator.mv(ydiff)) ** 2

    return squared_scaled_deviation


def _make_objective(y, constraint, penalty_parameter):
    squared_scaled_deviation = _make_scaled_deviation(y)

    def objective(_y, args):
        del args
        residual = constraint(_y)  # TODO: Use least-squares algorithm?
        # residual = jnp.where(residual < 0, -residual, 0)  # Only for inequalities!
        equality_residual, inequality_residual = residual
        if inequality_residual is not None:
            inequality_residual = jtu.tree_map(
                lambda x: jnp.where(x < 0, x, 0), inequality_residual
            )
        # TODO: enable usage of other kinds of norms
        residuals = (equality_residual, inequality_residual)
        return two_norm(residuals) + penalty_parameter * squared_scaled_deviation(_y)

    return objective


class ClosestFeasiblePoint(AbstractBoundaryMap, strict=True):
    """Find the closest point on a feasible set defined by the constraint function. The
    feasible set needs not be convex, in which case the projection onto the set is
    generally not unique. In this map, this is compensated for by penalizing deviation
    from the original point. If the feasible set is convex, the projection is unique and
    the penalty parameter can be set to zero.
    """

    penalty_parameter: ScalarLike = eqx.field(converter=scalarlike_asarray)
    solver: AbstractMinimiser

    # TODO: check invariants if this thing survives. I'm currently sceptical that it
    # will make sense to keep this boundary map pattern around in its current form, so
    # fixing this is not a priority - I don't know why it failed though, a very similar
    # pattern works for BacktrackingArmijo.
    # def __post_init__(self):
    #     self.penalty_parameter = eqx.error_if(
    #         self.penalty_parameter,
    #         self.penalty_parameter < jnp.array(0),
    #         "`ClosestFeasiblePoint(penalty_parameter=...)` must be non-negative.",
    #     )

    def __call__(
        self,
        y: Y,
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
    ) -> tuple[Y, RESULTS]:
        if constraint is None:
            raise ValueError("ClosestFeasiblePoint projection requires a constraint.")
        else:
            objective = _make_objective(y, constraint, self.penalty_parameter)
            sol = minimise(
                objective,
                self.solver,
                y,
                bounds=bounds,
                throw=False,
            )
            # sol=quadratic_solve(objective, self.solver, y, bounds=bounds, throw=False)
            return sol.value, sol.result
