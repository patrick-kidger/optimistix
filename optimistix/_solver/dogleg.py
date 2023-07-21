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

from collections.abc import Callable
from typing import Any

import jax.lax as lax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from .._custom_types import Y
from .._descent import AbstractDescent, AbstractLineSearch
from .._misc import (
    max_norm,
    sum_squares,
    tree_dot,
    tree_full_like,
    tree_where,
    two_norm,
)
from .._root_find import AbstractRootFinder, root_find
from .._solution import RESULTS
from .bisection import Bisection
from .gauss_newton import AbstractGaussNewton
from .trust_region import ClassicalTrustRegion


class DoglegDescent(AbstractDescent[Y]):
    """The Dogleg descent step, which switches between the Cauchy and the Newton
    descent directions.

    This requires the following `options`:

    - `vector`: The residual vector if `gauss_newton=True`, the gradient vector
        otherwise.
    - `operator`: The Jacobian operator of a least-squares problem if
        `gauss_newton=True`, the approximate Hessian of the objective function if not.
    """

    gauss_newton: bool
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    root_finder: AbstractRootFinder = Bisection(rtol=1e-3, atol=1e-3)
    trust_region_norm: Callable[[PyTree], Scalar] = two_norm

    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        vector = options["vector"]
        operator = options["operator"]
        if self.gauss_newton:
            # Compute the normalization in the gradient direction:
            # `g^T g (g^T B g)^(-1)` where `g` is `J^T r` (Jac and residual) and
            # `B` is `J^T J.`
            operator_transpose = operator.transpose()
            grad = operator_transpose.mv(vector)
            mvp = operator_transpose.mv(operator.mv(grad))
        else:
            grad = vector
            mvp = operator.mv(grad)

        newton_soln = lx.linear_solve(operator, vector, solver=self.linear_solver)
        newton = (-newton_soln.value**ω).ω
        newton_norm = self.trust_region_norm(newton)

        #
        # For trust-region methods like `DoglegDescent`, the trust-region size directly
        # controls how large a step we take. This is actually somewhat annoying,
        # as a trust region algorithm has no understanding of the scale of a
        # problem unless initialised with a complex initialisation procedure.
        #
        # A simple, heuristic way around this is to scale the trust region `step_size`
        # so that a `step_size` of `1` corresponds to the full length of the Newton
        # step (anything greater than `1` will still accept the Newton step.)
        #
        scaled_step_size = newton_norm * step_size

        denom = tree_dot(grad, mvp)
        denom_nonzero = denom > jnp.finfo(denom.dtype).eps
        safe_denom = jnp.where(denom_nonzero, denom, 1)
        scaling = jnp.where(denom_nonzero, sum_squares(grad) / safe_denom, 0.0)
        cauchy = (-scaling * grad**ω).ω
        cauchy_norm = self.trust_region_norm(cauchy)

        def accept_cauchy_or_newton(cauchy, newton):
            """Scale and return the Cauchy or Newton step."""
            norm_nonzero = cauchy_norm > jnp.finfo(cauchy_norm.dtype).eps
            safe_norm = jnp.where(norm_nonzero, cauchy_norm, 1)
            # Return zeros in degenerate case instead of inf because if `cauchy_norm` is
            # near zero, then so is the gradient and `delta` must be tiny to accept
            # this.
            normalised_cauchy = tree_where(
                norm_nonzero,
                ((cauchy**ω / safe_norm) * scaled_step_size).ω,
                tree_full_like(cauchy, 0),
            )
            return tree_where(cauchy_norm > scaled_step_size, normalised_cauchy, newton)

        def interpolate_cauchy_and_newton(cauchy, newton):
            """Find the point interpolating the Cauchy and Newton steps which
            intersects the trust region radius.
            """

            def interpolate(t):
                return (cauchy**ω + (t - 1) * (newton**ω - cauchy**ω)).ω

            # The vast majority of the time we expect users to use `two_norm`,
            # ie. the classic, elliptical trust region radius. In this case, we
            # compute the value of `t` to hit the trust region radius using by solving
            # a quadratic equation `a * t**2 + b * t + c = 0`
            # See section 4.1 of Nocedal Wright "Numerical Optimization" for details.
            #
            # If they pass a norm other than `two_norm`, ie. they use a more exotic
            # trust region shape, we use a root find to approximately
            # find the value which hits the trust region radius.
            if self.trust_region_norm == two_norm:
                a = sum_squares((newton**ω - cauchy**ω).ω)
                inner_prod = tree_dot(cauchy, (newton**ω - cauchy**ω).ω)
                b = 2 * (inner_prod - a)
                c = cauchy_norm**2 - 2 * inner_prod + a - scaled_step_size**2
                quadratic_1 = jnp.clip(
                    0.5 * (-b + jnp.sqrt(b**2 - 4 * a * c)) / a, a_min=1, a_max=2
                )
                quadratic_2 = jnp.clip(
                    ((2 * c) / (-b - jnp.sqrt(b**2 - 4 * a * c))), a_min=1, a_max=2
                )
                # The quadratic formula is not numerically stable, and it is best to
                # use slightly different formulas when `b >=` and `b < 0`.
                # See https://github.com/fortran-lang/stdlib/issues/674 for a number of
                # references.
                interp_amount = jnp.where(b >= 0, quadratic_1, quadratic_2)
            else:
                root_find_options = {"lower": jnp.array(1.0), "upper": jnp.array(2.0)}
                interp_amount = root_find(
                    lambda t, args: self.trust_region_norm(interpolate(t))
                    - scaled_step_size,
                    self.root_finder,
                    y0=1.5,
                    options=root_find_options,
                    throw=False,
                ).value
            return interpolate(interp_amount)

        diff = lax.cond(
            (cauchy_norm > scaled_step_size) | (step_size >= 1),
            accept_cauchy_or_newton,
            interpolate_cauchy_and_newton,
            cauchy,
            newton,
        )
        return diff, RESULTS.promote(newton_soln.result)


DoglegDescent.__init__.__doc__ = """**Arguments:**

- `gauss_newton`: `True` if this is used for a least squares problem, `False`
    otherwise.
- `linear_solver`: The linear solver used to compute the Newton step.
- `root_finder`: The root finder used to find the point where the trust-region
    intersects the dogleg path. This is ignored if
    `trust_region_norm=optimistix.two_norm`, for which there is an analytic formula
    instead.
- `trust_region_norm`: The norm used to determine the trust-region shape.
"""


class Dogleg(AbstractGaussNewton):
    """Dogleg algorithm. Used for nonlinear least squares problems.

    Given a quadratic bowl that locally approximates the function to be minimised, then
    there are two different ways we might try to move downhill: in the steepest descent
    direction (as in gradient descent; this is also sometimes called the Cauchy
    direction), and in the direction of the minima of the quadratic bowl (as in Newton's
    method; correspondingly this is called the Newton direction).

    The distinguishing feature of this algorithm is the "dog leg" shape of its descent
    path, in which it begins by moving in the steepest descent direction, and then
    switches to moving in the Newton direction.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    line_search: AbstractLineSearch

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
    ):
        # NOTE: we don't expose root_finder to the defaul API for Dogleg because
        # we assume the `trust_region_norm` norm is `two_norm`, which has
        # an analytic formula for the intersection with the dogleg path.
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.line_search = ClassicalTrustRegion(
            DoglegDescent(gauss_newton=True, linear_solver=linear_solver),
            gauss_newton=True,
        )


Dogleg.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `linear_solver`: The linear solver used to compute the Newton step.
"""
