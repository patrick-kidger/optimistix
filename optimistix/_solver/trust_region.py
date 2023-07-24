# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, Generic, Optional, TYPE_CHECKING

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar, ScalarLike


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar

from .._custom_types import DescentState, NoAuxFn, Out, Y
from .._misc import (
    sum_squares,
    tree_dot,
    tree_full_like,
    tree_where,
)
from .._search import AbstractDescent, AbstractSearch, DerivativeInfo
from .._solution import RESULTS


class _TrustRegionState(eqx.Module, Generic[DescentState]):
    step_size: Scalar
    acceptable: Bool[Array, ""]
    descent_state: DescentState


class _AbstractTrustRegion(AbstractSearch[Y, Out, _TrustRegionState]):
    """The abstract base class of the trust-region update algorithm.

    Trust region line searches compute the ratio
    `true_reduction/predicted_reduction`, where `true_reduction` is the decrease in `fn`
    between `y` and `new_y`, and `predicted_reduction` is how much we expected the
    function to decrease using an approximation to `fn`.

    The trust-region ratio determines whether to accept or reject a step and the
    next choice of step-size. Specifically:

    - reject the step and decrease stepsize if the ratio is smaller than a
        cutoff `low_cutoff`
    - accept the step and increase the step-size if the ratio is greater than
        another cutoff `high_cutoff` with `low_cutoff < high_cutoff`.
    - else, accept the step and make no change to the step-size.
    """

    high_cutoff: AbstractVar[ScalarLike]
    low_cutoff: AbstractVar[ScalarLike]
    high_constant: AbstractVar[ScalarLike]
    low_constant: AbstractVar[ScalarLike]

    def __post_init__(self):
        # You would not expect `self.low_cutoff` or `self.high_cutoff` to
        # be below zero, but this is technically not incorrect so we don't
        # require it.
        self.low_cutoff, self.high_cutoff = eqx.error_if(  # pyright: ignore
            (self.low_cutoff, self.high_cutoff),
            self.low_cutoff > self.high_cutoff,  # pyright: ignore
            "`low_cutoff` must be below `high_cutoff` in `ClassicalTrustRegion`",
        )
        self.low_constant = eqx.error_if(  # pyright: ignore
            self.low_constant,
            self.low_constant < 0,  # pyright: ignore
            "`low_constant` must be greater than `0` in `ClassicalTrustRegion`",
        )
        self.high_constant = eqx.error_if(  # pyright: ignore
            self.high_constant,
            self.high_constant < 0,  # pyright: ignore
            "`high_constant` must be greater than `0` in `ClassicalTrustRegion`",
        )

    @abc.abstractmethod
    def predict_reduction(self, deriv_info: DerivativeInfo, y_diff: Y) -> Scalar:
        ...

    def init(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Scalar],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> _TrustRegionState:
        return _TrustRegionState(
            step_size=jnp.array(1.0),
            acceptable=jnp.array(True),
            descent_state=descent.optim_init(fn, y, args, f_struct),
        )

    def search(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: _TrustRegionState,
        deriv_info: DerivativeInfo,
        max_steps: Optional[int],
    ) -> tuple[Y, Bool[Array, ""], RESULTS, _TrustRegionState]:
        def _cache_descent_state():
            return state.descent_state

        def _compute_descent_state():
            return descent.search_init(fn, y, args, f, state.descent_state, deriv_info)

        descent_state = lax.cond(
            eqxi.unvmap_any(state.acceptable),
            _compute_descent_state,
            _cache_descent_state,
        )
        y_diff, result, descent_state = descent.descend(
            state.step_size, fn, y, args, f, descent_state, deriv_info
        )

        if isinstance(
            deriv_info,
            (
                DerivativeInfo.Grad,
                DerivativeInfo.GradHessian,
                DerivativeInfo.GradHessianInv,
            ),
        ):
            min_fn = fn
            min_f0 = f
        elif isinstance(deriv_info, DerivativeInfo.ResidualJac):
            min_fn = lambda y, args: 0.5 * sum_squares(fn(y, args))
            min_f0 = 0.5 * sum_squares(f)
        else:
            assert False
        assert isinstance(min_f0, Scalar)

        y_candidate = (y**ω + y_diff**ω).ω
        min_f = min_fn(y_candidate, args)
        predicted_reduction = self.predict_reduction(deriv_info, y_diff)
        f_diff = min_f - min_f0
        # We never actually compute the ratio
        # `true_reduction/predicted_reduction`. Instead, we rewrite the conditions as
        # `true_reduction < const * predicted_reduction` instead, where the inequality
        # flips because we assume `predicted_reduction` is negative.
        # This avoids an expensive division.
        acceptable = f_diff <= self.low_cutoff * predicted_reduction
        good = f_diff < self.high_cutoff * predicted_reduction
        acceptable = acceptable & (predicted_reduction <= 0)
        good = good & (predicted_reduction < 0)
        mul = jnp.where(good, self.high_constant, 1)
        mul = jnp.where(acceptable, mul, self.low_constant)
        new_step_size = mul * state.step_size
        new_step_size = jnp.where(
            new_step_size < jnp.finfo(new_step_size.dtype).eps,
            jnp.array(1.0),
            new_step_size,
        )
        y_diff = tree_where(acceptable, y_diff, tree_full_like(y, 0))
        new_state = _TrustRegionState(
            step_size=new_step_size,
            acceptable=acceptable,
            descent_state=descent_state,
        )
        # TODO(kidger): there are two more optimisations we can make here.
        # (1) if we reject the step, then we can avoid recomputing f0 and grad on the
        #     next iteration of the outer solver. This has other downstream impacts:
        #     it means deriv_info is unchanged, which in turn means we can avoid
        #     recomputing the init part of newton_step.
        # (2) if we accept the step, then we can avoid recomputing f0 on the next
        #     iteration of the outer solver.
        return y_diff, acceptable, result, new_state


class ClassicalTrustRegion(_AbstractTrustRegion[Y, Out]):
    """The classic trust-region update algorithm which uses a quadratic approximation of
    the objective function to predict reduction.

    Building a quadratic approximation requires an approximation to the Hessian of the
    overall minimisation function. This means that trust region is suitable for use with
    least-squares algorithms (which make the Gauss--Newton approximation
    Hessian~Jac^T J) and for quasi-Newton minimisation algorithms like
    [`optimistix.BFGS`][]. (An error will be raised if you use this with an incompatible
    solver.)
    """

    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."
    high_cutoff: ScalarLike = 0.99
    low_cutoff: ScalarLike = 0.01
    high_constant: ScalarLike = 3.5
    low_constant: ScalarLike = 0.25

    def predict_reduction(self, deriv_info: DerivativeInfo, y_diff: Y) -> Scalar:
        """Compute the expected decrease in loss from taking the step `y_diff`.

        The true reduction is
        ```
        fn(y0 + y_diff) - fn(y0)
        ```
        so if `B` is the approximation to the Hessian coming from the quasi-Newton
        method at `y`, and `g` is the gradient of `fn` at `y`, then the predicted
        reduction is
        ```
        g^T y_diff + 1/2 y_diff^T B y_diff
        ```

        **Arguments**:

        - `deriv_info`: the derivative information (on the gradient and Hessian)
            provided by the outer loop.
        - `y_diff`: the proposed step by the descent method.

        **Returns**:

        The expected decrease in loss from moving from `y0` to `y0 + y_diff`.
        """

        if isinstance(deriv_info, (DerivativeInfo.Grad, DerivativeInfo.GradHessianInv)):
            raise ValueError(
                "Cannot use `ClassicalTrustRegion` with this solver. This is because "
                "`ClassicalTrustRegion` requires (an approximation to) the Hessian of "
                "the target function, but this solver does not make any estimate of "
                "that information."
            )
        elif isinstance(deriv_info, DerivativeInfo.GradHessian):
            # Minimisation algorithm. Directly compute the quadratic approximation.
            return tree_dot(
                y_diff,
                (deriv_info.grad**ω + 0.5 * deriv_info.hessian.mv(y_diff) ** ω).ω,
            )
        elif isinstance(deriv_info, DerivativeInfo.ResidualJac):
            # Least-squares algorithm. So instead of considering fn (which returns the
            # residuals), instead consider `0.5*fn(y)^2`, and then apply the logic as
            # for minimisation.
            # We get that `g = J^T f0` and `B = J^T J + dJ/dx^T J`.
            # (Here, `f0 = fn(y0)` are the residuals, and `J = dfn/dy(y0)` is the
            # Jacobian of the residuals wrt y.)
            # Then neglect the second term in B (the usual Gauss--Newton approximation)
            # and complete the square.
            # We find that the predicted reduction is
            # `0.5 * ((J y_diff + f0)^T (J y_diff + f0) - f0^T f0)`
            # and this is what is written below.
            #
            # The reason we go through this hassle is because this now involves only
            # a single Jacobian-vector product, rather than the three we would have to
            # make by naively substituting `B = J^T J `and `g = J^T f0` into the general
            # algorithm used for minimisation.
            rtr = sum_squares(deriv_info.residual)
            jacobian_term = sum_squares(
                (deriv_info.jac.mv(y_diff) ** ω + deriv_info.residual**ω).ω
            )
            return 0.5 * (jacobian_term - rtr)
        else:
            assert False


# When using a gradient-based method, `LinearTrustRegion` is actually a variant of
# `BacktrackingArmijo`. The linear predicted reduction is the same as the Armijo
# condition. The difference is that unlike standard backtracking,
# `LinearTrustRegion` chooses its next step size based on how well it did in the
# previous iteration.
class LinearTrustRegion(_AbstractTrustRegion[Y, Out]):
    """The trust-region update algorithm which uses a linear approximation of
    the objective function to predict reduction.

    Generally speaking you should prefer [`optimistix.ClassicalTrustRegion`][], unless
    you happen to be using a solver (e.g. a non-quasi-Newton minimiser) with which that
    is incompatible.
    """

    # This choice of default parameters comes from Gould et al.
    # "Sensitivity of trust region algorithms to their parameters."
    high_cutoff: ScalarLike = 0.99
    low_cutoff: ScalarLike = 0.01
    high_constant: ScalarLike = 3.5
    low_constant: ScalarLike = 0.25

    def predict_reduction(self, deriv_info: DerivativeInfo, y_diff: Y) -> Scalar:
        """Compute the expected decrease in loss from taking the step `y_diff`.

        The true reduction is
        ```
        fn(y0 + y_diff) - fn(y0)
        ```
        so if `g` is the gradient of `fn` at `y`, then the predicted reduction is
        ```
        g^T y_diff
        ```

        **Arguments**:

        - `deriv_info`: the derivative information (on the gradient and Hessian)
            provided by the outer loop.
        - `y_diff`: the proposed step by the descent method.

        **Returns**:

        The expected decrease in loss from moving from `y0` to `y0 + y_diff`.
        """

        assert isinstance(
            deriv_info,
            (
                DerivativeInfo.Grad,
                DerivativeInfo.GradHessian,
                DerivativeInfo.GradHessian,
                DerivativeInfo.ResidualJac,
            ),
        )
        return tree_dot(deriv_info.grad, y_diff)


_init_doc = """In the following, `ratio` refers to the ratio
`true_reduction/predicted_reduction`.

**Arguments**:

- `high_cutoff`: the cutoff such that `ratio > high_cutoff` will accept the step
and increase the step-size on the next iteration.
- `low_cutoff`: the cutoff such that `ratio < low_cutoff` will reject the step
and decrease the step-size on the next iteration.
- `high_constant`: when `ratio > high_cutoff`, multiply the previous step-size by
high_constant`.
- `low_constant`: when `ratio < low_cutoff`, multiply the previous step-size by
low_constant`.
"""

LinearTrustRegion.__init__.__doc__ = _init_doc
ClassicalTrustRegion.__init__.__doc__ = _init_doc
