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
from .._descent import AbstractDescent
from .._misc import (
    max_norm,
    sum_squares,
    tree_full_like,
    tree_inner_prod,
    tree_where,
    two_norm,
)
from .._root_find import AbstractRootFinder, root_find
from .._solution import RESULTS
from .bisection import Bisection
from .gauss_newton import AbstractGaussNewton
from .trust_region import ClassicalTrustRegion


class DoglegDescent(AbstractDescent[Y]):
    gauss_newton: bool
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    root_finder: AbstractRootFinder = Bisection(rtol=1e-3, atol=1e-3)
    trust_region_norm: Callable = two_norm

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

        denom = tree_inner_prod(grad, mvp)
        denom_nonzero = denom > jnp.finfo(denom.dtype).eps
        safe_denom = jnp.where(denom_nonzero, denom, 1)
        scaling = jnp.where(denom_nonzero, sum_squares(grad) / safe_denom, 0.0)
        cauchy = (-scaling * grad**ω).ω
        cauchy_norm = self.trust_region_norm(cauchy)

        def accept_cauchy_or_newton(cauchy_newton):
            """Scale and return the Cauchy or Newton step."""
            cauchy, newton = cauchy_newton
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

        def interpolate_cauchy_and_newton(cauchy_newton):
            """Find the point interpolating the Cauchy and Newton steps which
            intersects the trust region radius.
            """
            cauchy, newton = cauchy_newton

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
                inner_prod = tree_inner_prod(cauchy, (newton**ω - cauchy**ω).ω)
                b = 2 * (inner_prod - a)
                c = cauchy_norm**2 - 2 * inner_prod + a - scaled_step_size**2
                quadratic_1 = jnp.clip(
                    0.5 * (-b + jnp.sqrt(b**2 - 4 * a * c)) / a, a_min=1, a_max=2
                )
                quadratic_2 = jnp.clip(
                    ((2 * c) / (-b - jnp.sqrt(b**2 - 4 * a * c))), a_min=1, a_max=2
                )
                # The quadratic formula is not numerically stable, and it is best to
                # use slightly different formulas when `b >=` and `b < 0` to avoid
                # See this issue: https://github.com/fortran-lang/stdlib/issues/674
                # for a number of references.
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

        # Pass and unpack the tuple `(cauchy_dynamic, newton_dynamic)` rather than
        # directly pass `cauchy_dynamic` and `newton_dynamic` to work around JAX
        # issue #16413
        diff = lax.cond(
            (cauchy_norm > scaled_step_size) | (step_size >= 1),
            accept_cauchy_or_newton,
            interpolate_cauchy_and_newton,
            (cauchy, newton),
        )
        return diff, RESULTS.promote(newton_soln.result)


class Dogleg(AbstractGaussNewton):
    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable = max_norm,
        root_finder: AbstractRootFinder = Bisection(rtol=0.001, atol=0.001),
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.line_search = ClassicalTrustRegion(
            DoglegDescent(
                gauss_newton=True, linear_solver=linear_solver, root_finder=root_finder
            ),
            gauss_newton=True,
        )
