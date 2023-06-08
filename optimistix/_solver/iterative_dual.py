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

from typing import (
    Any,
    Callable,
    Optional,
)

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._line_search import AbstractDescent
from .._misc import max_norm, tree_where
from .._root_find import AbstractRootFinder, root_find
from .._solution import RESULTS
from .misc import quadratic_predicted_reduction
from .newton_chord import Newton


#
# NOTE: This method is usually called Levenberg-Marquardt. However,
# Levenberg-Marquard often refers specifically to the case where this approach
# is applied in the Gauss-Newton setting. For this reason, we refer to the approach
# by the more generic name "iterative dual."
#
# Iterative dual is a method of solving for the descent direction given a
# trust region radius. It does this by solving the dual problem
# `(B + lambda I) p = r` for `p`, where `B` is the quasi-Newton matrix,
# lambda is the dual parameter (the dual parameterisation of the
# trust region radius), `I` is the identity, and `r` is the vector of residuals.
#
# Iterative dual is approached in one of two ways:
# 1. set the trust region radius and find the Levenberg-Marquadt parameter
# `lambda` which will give approximately solve the trust region problem. ie.
# solve the dual of the trust region problem.
#
# 2. set the Levenberg-Marquadt parameter `lambda` directly.
#
# Respectively, this is the indirect and direct approach to iterative dual.
# The direct approach is common in practice, and is often interpreted as
# interpolating between quasi-Newton and gradient based approaches.
#


class _IterativeDualState(eqx.Module):
    y: PyTree[Array]
    fn: Fn[Scalar, Scalar, Any]
    vector: PyTree[Array]
    operator: lx.AbstractLinearOperator


class _Damped(eqx.Module):
    fn: Callable
    damping: Float[Array, " "]

    def __call__(self, y: PyTree[Array], args: Any):
        damping = jnp.sqrt(self.damping)
        f, aux = self.fn(y, args)
        damped = jtu.tree_map(lambda yi: damping * yi, y)
        return (f, damped), aux


class _DirectIterativeDual(AbstractDescent[_IterativeDualState]):
    gauss_newton: bool

    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: Any,
        options: dict[str, Any],
    ) -> _IterativeDualState:
        if operator is None:
            assert False
        return _IterativeDualState(y, fn, vector, operator)

    def update_state(
        self,
        descent_state: _IterativeDualState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ) -> _IterativeDualState:
        return _IterativeDualState(descent_state.y, descent_state.fn, vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        if self.gauss_newton:
            vector = (descent_state.vector, ω(descent_state.y).call(jnp.zeros_like).ω)
            operator = lx.JacobianLinearOperator(
                _Damped(descent_state.fn, delta),
                descent_state.y,
                args,
                _has_aux=True,
            )
        else:
            vector = descent_state.vector
            operator = descent_state.operator + delta * lx.IdentityLinearOperator(
                descent_state.operator.in_structure()
            )
        operator = lx.linearise(operator)
        linear_soln = lx.linear_solve(operator, vector, lx.QR(), throw=False)
        diff = (-linear_soln.value**ω).ω
        return diff, RESULTS.promote(linear_soln.result)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ) -> Scalar:
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )


class DirectIterativeDual(_DirectIterativeDual):
    def __call__(
        self,
        delta: Scalar,
        descent_state: _IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        if descent_state.operator is None:
            raise ValueError(
                "`operator` must be passed to `DirectIterativeDual`. "
                "Note that `operator_inv` is not currently supported for this descent."
            )

        delta_nonzero = delta > jnp.finfo(delta.dtype).eps
        if self.gauss_newton:
            vector = (descent_state.vector, ω(descent_state.y).call(jnp.zeros_like).ω)
            operator = lx.JacobianLinearOperator(
                _Damped(
                    descent_state.fn,
                    jnp.where(delta_nonzero, 1 / delta, jnp.inf),
                ),
                descent_state.y,
                args,
                _has_aux=True,
            )
        else:
            vector = descent_state.vector
            operator = descent_state.operator + jnp.where(
                delta_nonzero, 1 / delta, jnp.inf
            ) * lx.IdentityLinearOperator(descent_state.operator.in_structure())
        operator = lx.linearise(operator)
        linear_soln = lx.linear_solve(operator, vector, lx.QR(), throw=False)
        no_diff = jtu.tree_map(jnp.zeros_like, linear_soln.value)
        diff = tree_where(delta_nonzero, (-linear_soln.value**ω).ω, no_diff)
        return diff, RESULTS.promote(linear_soln.result)


class IndirectIterativeDual(AbstractDescent[_IterativeDualState]):
    gauss_newton: bool
    lambda_0: Float[Array, ""]
    root_finder: AbstractRootFinder = Newton(rtol=1e-2, atol=1e-2, lower=1e-5)
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)
    tr_reg: Optional[lx.PyTreeLinearOperator] = None
    norm: Callable = max_norm

    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: Any,
        options: dict[str, Any],
    ) -> _IterativeDualState:
        if operator is None:
            assert False
        return _IterativeDualState(y, fn, vector, operator)

    def update_state(
        self,
        descent_state: _IterativeDualState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ):
        return _IterativeDualState(descent_state.y, descent_state.fn, vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        if descent_state.operator is None:
            raise ValueError(
                "`operator` must be passed to "
                " `IndirectDirectIterativeDual`. Note that `operator_inv` is "
                "not currently supported for this descent."
            )

        direct_dual = eqx.Partial(
            _DirectIterativeDual(self.gauss_newton),
            descent_state=descent_state,
            args=args,
            options=options,
        )
        newton_soln = lx.linear_solve(
            descent_state.operator,
            (-descent_state.vector**ω).ω,
            self.linear_solver,
            throw=False,
        )
        # NOTE: try delta = delta * self.norm(newton_step).
        # this scales the trust and sets the natural bound `delta = 1`.
        newton_step = (-ω(newton_soln.value)).ω
        newton_result = RESULTS.promote(newton_soln.result)
        tr_reg = self.tr_reg

        if tr_reg is None:
            tr_reg = lx.IdentityLinearOperator(jax.eval_shape(lambda: newton_step))

        def comparison_fn(
            lambda_i: Scalar,
            args: Any,
        ):
            (step, _) = direct_dual(lambda_i)
            return self.norm(step) - delta

        def accept_newton():
            return newton_step, newton_result

        def reject_newton():
            root_find_options = {
                "vector": descent_state.vector,
                "operator": descent_state.operator,
                "delta": delta,
            }
            lambda_out = root_find(
                fn=comparison_fn,
                has_aux=False,
                solver=self.root_finder,
                y0=self.lambda_0,
                args=args,
                options=root_find_options,
                max_steps=32,
                throw=False,
            ).value
            return direct_dual(lambda_out)

        newton_norm = self.norm(newton_step)
        return lax.cond(newton_norm < delta, accept_newton, reject_newton)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ) -> Scalar:
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )
