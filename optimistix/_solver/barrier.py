import abc
from typing import Generic

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Scalar, ScalarLike

from .._custom_types import Y
from .._misc import scalarlike_asarray, tree_where


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
    barrier = optx.LogarithmicBarrier(bounds, 1e-3)

    def fn(y, args):
        return f(y, args) + barrier(y)

    sol = optx.minimise(fn, unconstrained_solver, ...)
    ```

    Solvers such as [`optimistix.IPOPTLike`][] use barrier functions internally, and
    adaptively update the barrier parameter.
    """

    @abc.abstractmethod
    def __call__(self, y: Y) -> Scalar:
        """Call the barrier function. Returns a scalar value that to be added to the
        unconstrained objective.

        **Arguments:**
        - `y`: The current point in the search space.
        """


class LogarithmicBarrier(AbstractBarrier[Y], strict=True):
    """Implements a logarithmic barrier function, defined as the negative sum of the
    logarithm of the distances to the bounds, multiplied by the barrier parameter.

    Used in [`optimistix.IPOPTLike`][] to enforce bounds on the variables.
    """

    bounds: tuple[Y, Y]
    barrier_parameter: ScalarLike = eqx.field(converter=scalarlike_asarray)

    def __call__(self, y: Y) -> Scalar:
        lower, upper = self.bounds

        # Take the logarithm of the positive distance to finite bounds
        lower_dist = jtu.tree_map(jnp.log, (y**ω - lower**ω).ω)
        upper_dist = jtu.tree_map(jnp.log, (upper**ω - y**ω).ω)
        safe_lower_dist = tree_where(jtu.tree_map(jnp.isfinite, lower), lower_dist, 0.0)
        safe_upper_dist = tree_where(jtu.tree_map(jnp.isfinite, upper), upper_dist, 0.0)

        lower_contribution, _ = jfu.ravel_pytree(safe_lower_dist)
        upper_contribution, _ = jfu.ravel_pytree(safe_upper_dist)
        summed_contributions = jnp.sum(lower_contribution) + jnp.sum(upper_contribution)

        return -jnp.asarray(self.barrier_parameter) * summed_contributions

    # TODO: gradient should be an abstract method, we will always need this
    # Add to documentation that the gradients are expected to be additive
    def grad(self, y: Y) -> tuple[Y, Y]:
        lower, upper = self.bounds

        lower_dist = (1 / (lower**ω - y**ω)).ω  # Flip order to get a negative sign
        upper_dist = (1 / (upper**ω - y**ω)).ω
        safe_lower_dist = tree_where(jtu.tree_map(jnp.isfinite, lower), lower_dist, 0.0)
        safe_upper_dist = tree_where(jtu.tree_map(jnp.isfinite, upper), upper_dist, 0.0)

        lower_gradient = (jnp.asarray(self.barrier_parameter) * safe_lower_dist**ω).ω
        upper_gradient = (jnp.asarray(self.barrier_parameter) * safe_upper_dist**ω).ω
        return lower_gradient, upper_gradient
