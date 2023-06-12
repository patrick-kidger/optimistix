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

from typing import Any, Callable, Optional

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._line_search import AbstractDescent
from .._misc import sum_squares, tree_inner_prod, tree_where, tree_zeros_like, two_norm
from .._solution import RESULTS
from .misc import quadratic_predicted_reduction


def _quadratic_solve(a: Scalar, b: Scalar, c: Scalar) -> Scalar:
    # If the output is >2 then we accpet the Newton step.
    # If it is below 1 then we accept the Cauchy step.
    # This clip is just to keep us within those theoretical bounds.
    return jnp.clip(0.5 * (-b + jnp.sqrt(b**2 - 4 * a * c)) / a, a_min=1, a_max=2)


class _DoglegState(eqx.Module):
    vector: PyTree[Array]
    operator: lx.AbstractLinearOperator


# `norm` has nothing to do with convergence here, unlike in our solvers. There is no
# notion of termination/convergence for `Dogleg`. Instead, this controls
# the shape of the trust region. The default of `two_norm` is standard.
class Dogleg(AbstractDescent[_DoglegState]):
    gauss_newton: bool
    norm: Callable = two_norm
    solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)

    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: PyTree,
        options: dict[str, Any],
    ) -> _DoglegState:
        if operator is None:
            assert False
        return _DoglegState(vector, operator)

    def update_state(
        self,
        descent_state: _DoglegState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ) -> _DoglegState:
        return _DoglegState(vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _DoglegState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:

        if self.gauss_newton:
            # Compute the normalization in the gradient direction: g^T g (g^T B g)^(-1)
            # where g is J^T r (Jac and residual) and B is J^T J.
            grad = descent_state.operator.transpose().mv(descent_state.vector)
            mvp = descent_state.operator.transpose().mv(descent_state.operator.mv(grad))
        else:
            grad = descent_state.vector
            mvp = descent_state.operator.mv(grad)

        numerator = sum_squares(grad)
        denominator = tree_inner_prod(grad, mvp)
        pred = denominator > jnp.finfo(denominator.dtype).eps
        safe_denom = jnp.where(pred, denominator, 1)
        projection_const = jnp.where(pred, numerator / safe_denom, jnp.inf)
        # Compute Newton and Cauchy steps. If below Cauchy or above Newton,
        # accept (scaled) cauchy or Newton respectively.
        cauchy = (-projection_const * grad**ω).ω
        newton_soln = lx.linear_solve(
            descent_state.operator, descent_state.vector, solver=self.solver
        )
        newton = (-newton_soln.value**ω).ω
        cauchy_norm = self.norm(cauchy)
        newton_norm = self.norm(newton)
        above_newton = newton_norm <= delta
        below_cauchy = cauchy_norm > delta
        between_cauchy_and_newton = jnp.invert(above_newton) & jnp.invert(below_cauchy)
        # If between the cauchy and newton step, interpolate between them.
        # We can calculate the exact interpolating scalar `τ` by solving a quadratic
        # equation `a * τ**2 + b * τ + c = 0`, hence the names of the variables.
        # See section 4.1 of Nocedal Wright "Numerical Optimization" for details.
        a = sum_squares((newton**ω - cauchy**ω).ω)
        b = 2 * tree_inner_prod(cauchy, (newton**ω - cauchy**ω).ω)
        c = cauchy_norm**2 - b - a - delta**2
        linear_interp = _quadratic_solve(a, b, c)
        dogleg = (cauchy**ω + (linear_interp - 1) * (newton**ω - cauchy**ω)).ω
        norm_nonzero = cauchy_norm > jnp.finfo(cauchy_norm.dtype).eps
        safe_norm = jnp.where(norm_nonzero, cauchy_norm, 1)
        # Return zeros instead of inf because if cauchy norm is near `0`, then so
        # is the gradient and `delta` must be tiny to be return the cauchy step
        # so we just assume it's 0.
        normalised_cauchy = tree_where(
            norm_nonzero,
            ((cauchy**ω / safe_norm) * delta).ω,
            tree_zeros_like(cauchy),
        )
        diff = tree_where(below_cauchy, normalised_cauchy, newton)
        diff = tree_where(between_cauchy_and_newton, dogleg, diff)
        diff = tree_where(above_newton, newton, diff)
        return diff, RESULTS.promote(newton_soln.result)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _DoglegState,
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )
