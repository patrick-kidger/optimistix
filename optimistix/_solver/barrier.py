import abc
from typing import Generic

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Scalar, ScalarLike

from .._custom_types import Y
from .._misc import tree_where


# TODO: Potentially support custom JVPs for the barrier function? Most gradients should
# be trivial to compute though - for now this is relevant in the case where the user
# would use the Barrier class to convert the problem to an unconstrained problem. Since
# we need the gradient of `fn` separately, we're not currently using it in this way in
# optx.IPOPTLike.
class AbstractBarrier(eqx.Module, Generic[Y], strict=True):
    """Abstract base class for barrier functions. Barrier functions are functions of `y`
    whose output is a scalar quantity that is added to the value of the objective
    function `fn(y, args)`. Generally, this is done to assign a penalty to the objective
    function whenever `y` approaches its bounds.

    Bounds must be specified as a pair (`lower, upper`), where `lower` and `upper` have
    the same PyTree structure as `y`. The barrier function can be used to convert a
    bound-constrained problem into an unconstrained problem, by adding the barrier term
    to the objective function `f`:

    ```python
    barrier = optx.LogarithmicBarrier(bounds)

    def fn(y, args):
        return f(y, args) + barrier(y, 1e-3)

    sol = optx.minimise(fn, unconstrained_solver, ...)
    ```

    Solvers such as [`optimistix.IPOPTLike`][] use barrier functions internally, and
    adaptively update the barrier parameter.
    """

    @abc.abstractmethod
    def __call__(self, y: Y, barrier_parameter: ScalarLike) -> Scalar:
        """Call the barrier function. Returns a scalar value that to be added to the
        unconstrained objective.

        **Arguments:**
        - `y`: The current point in the search space.
        - `barrier_parameter`: The current barrier parameter.
        """

    @abc.abstractmethod
    def grads(self, y: Y, barrier_parameter: ScalarLike) -> tuple[Y, Y]:
        """Compute the gradient of the barrier function with respect to `y`. Returns a
        tuple of gradients for the lower and upper bounds that can be added to the
        gradient of the objective function `fn(y, args)`.

        We generally require access to each gradient separately, for instance when
        reconstructing solutions for the bound multipliers from condensed KKT systems.

        **Arguments:**
        - `y`: The current point in the search space.
        - `barrier_parameter`: The current barrier parameter.
        """

    @abc.abstractmethod
    def primal_dual_hessians(
        self, y: Y, bound_multipliers: tuple[Y, Y]
    ) -> tuple[lx.DiagonalLinearOperator, lx.DiagonalLinearOperator]:
        """Compute the primal-dual Hessians of the barrier function with respect to `y`.
        In contrast to the primal Hessian - which is just the second derivative of the
        barrier function - the primal-dual Hessian is defined as the product of the
        multipliers of the bound constraints and the gradients of the barrier function.
        At optimality, the primal-dual Hessian should coincide with the primal Hessian.

        We generally require access to each Hessian separately, for instance when
        reconstructing solutions for the bound multipliers from condensed KKT systems.

        **Arguments:**
        - `y`: The current point in the search space.
        - `bound_multipliers`: The current bound multipliers.
        """


class LogarithmicBarrier(AbstractBarrier[Y], strict=True):
    """Implements a logarithmic barrier function, defined as the negative sum of the
    logarithm of the distances to the bounds, multiplied by the barrier parameter.

    Used in [`optimistix.IPOPTLike`][] to enforce bounds on the variables.
    """

    bounds: tuple[Y, Y]

    def __call__(self, y: Y, barrier_parameter: ScalarLike) -> Scalar:
        lower, upper = self.bounds

        lower_dist = jtu.tree_map(jnp.log, (y**ω - lower**ω).ω)
        upper_dist = jtu.tree_map(jnp.log, (upper**ω - y**ω).ω)
        safe_lower_dist = tree_where(jtu.tree_map(jnp.isfinite, lower), lower_dist, 0.0)
        safe_upper_dist = tree_where(jtu.tree_map(jnp.isfinite, upper), upper_dist, 0.0)

        lower_contribution, _ = jfu.ravel_pytree(safe_lower_dist)
        upper_contribution, _ = jfu.ravel_pytree(safe_upper_dist)
        summed_contributions = jnp.sum(lower_contribution) + jnp.sum(upper_contribution)

        return -jnp.asarray(barrier_parameter) * summed_contributions

    def grads(self, y: Y, barrier_parameter: ScalarLike) -> tuple[Y, Y]:
        lower, upper = self.bounds

        lower_dist = (1 / (lower**ω - y**ω)).ω  # This gradient keeps its negative sign
        upper_dist = (1 / (upper**ω - y**ω)).ω  # (This one does not.)
        safe_lower_dist = tree_where(jtu.tree_map(jnp.isfinite, lower), lower_dist, 0.0)
        safe_upper_dist = tree_where(jtu.tree_map(jnp.isfinite, upper), upper_dist, 0.0)

        lower_gradient = (barrier_parameter * safe_lower_dist**ω).ω
        upper_gradient = (barrier_parameter * safe_upper_dist**ω).ω
        return lower_gradient, upper_gradient

    def primal_dual_hessians(
        self, y: Y, bound_multipliers: tuple[Y, Y]
    ) -> tuple[lx.DiagonalLinearOperator, lx.DiagonalLinearOperator]:
        lower, upper = self.bounds

        lower_dist = (1 / (y**ω - lower**ω)).ω
        upper_dist = (1 / (upper**ω - y**ω)).ω
        safe_lower_dist = tree_where(jtu.tree_map(jnp.isfinite, lower), lower_dist, 0.0)
        safe_upper_dist = tree_where(jtu.tree_map(jnp.isfinite, upper), upper_dist, 0.0)

        lower_multiplier, upper_multiplier = bound_multipliers
        lower_hessian = (lower_multiplier * safe_lower_dist**ω).ω
        upper_hessian = (upper_multiplier * safe_upper_dist**ω).ω

        lower_hessian = lx.DiagonalLinearOperator(lower_hessian)
        upper_hessian = lx.DiagonalLinearOperator(upper_hessian)

        return lower_hessian, upper_hessian
