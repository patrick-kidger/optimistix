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
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._line_search import AbstractDescent
from .._misc import tree_where, two_norm
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
# `(B + λ I) p = -g` for `p`, where `B` is the quasi-Newton matrix,
# `λ` is a Lagrange multiplier (often called the "Levenberg-Marquardt parameter"),
# `I` is the identity, and `g` is the gradient. In the Gauss-Newton setting,
# `B = J^T J` and `g = J^T r` where `J` is the Jacobian and `r` the residual
# vector.
#
# Iterative dual is approached in one of two ways:
# 1. Set the trust region radius and find the Levenberg-Marquadt parameter
# `λ` which will approximately solve the trust region problem.
#
# 2. Set the Levenberg-Marquadt parameter `λ` directly.
#
# Respectively, this is the indirect and direct approach to iterative dual.
# The direct approach is common in practice, and can be interpreted as
# interpolating between quasi-Newton and gradient based approaches.
#


class _Damped(eqx.Module):
    fn: Callable
    damping: Float[Array, " "]

    def __call__(self, y: PyTree[Array], args: PyTree):
        damping = jnp.sqrt(self.damping)
        f, aux = self.fn(y, args)
        damped = jtu.tree_map(lambda yi: damping * yi, y)
        return (f, damped), aux


class _IterativeDualState(eqx.Module):
    y: PyTree[Array]
    fn: Fn[Scalar, Scalar, Any]
    vector: PyTree[Array]
    operator: lx.AbstractLinearOperator


class _DirectIterativeDual(AbstractDescent[_IterativeDualState]):
    gauss_newton: bool

    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: PyTree,
        options: dict[str, Any],
    ) -> _IterativeDualState:
        # TODO(raderj): support operator_inv.
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
        if operator is None:
            assert False
        return _IterativeDualState(descent_state.y, descent_state.fn, vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _IterativeDualState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        if self.gauss_newton:
            # In the Gauss-Newton case, rather than solve `(J^T J + λ I) p = -J^T r`
            # by computing`J^T J` and squaring the condition number of `J`, we can
            # compute `[J, jnp.sqrt(λ) I]^T p = [-r, 0]^T`.
            # This is a common numerical approach.
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
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )


#
# Note that the Levenberg-Marquard parameter of `_DirectIterativeDual` is
# parameterised by `delta`, but in `DirectIterativeDual` it is `1/delta`.
# It is easier to find the correct damping using a Newton solve with the former,
# but for line searches we usually assume that the smaller the value of `delta` the
# smaller the step that we take.
#
# In particular, if we do the eigendecomposition
# `p = -(B + λ I)^(-1) g = -U (Λ + λ I) U^T g`
# then the square of the two norm of `p` as a function of `λ` is
# ```
# ||p||**2 = sum [U g]_i/((λ_i + λ)**2)
# ```
# where `λ_i = Λ_i` is the ith eigenvector of `B` and `[U g]_i` is the `ith` element
# of the vector `U g`.
# From this we see that the parameterising `1/delta` can be viewed as
# approximately controlling the norm of `p`.
#
class DirectIterativeDual(_DirectIterativeDual):
    def __call__(
        self,
        delta: Scalar,
        descent_state: _IterativeDualState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
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


# `norm` has nothing to do with convergence here, unlike in our solvers. There is no
# notion of termination/convergence for `IndirectIterativeDual`. Instead, this controls
# the shape of the trust region. The default of `two_norm` is standard.
class IndirectIterativeDual(AbstractDescent[_IterativeDualState]):
    gauss_newton: bool
    lambda_0: Float[Array, ""]
    root_finder: AbstractRootFinder = Newton(rtol=1e-2, atol=1e-2, lower=1e-5)
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)
    norm: Callable = two_norm
    # Default tol for `root_finder` only because a user would override this entirely
    # with `FooRootFinder(rtol, atol, ...)` and the tol doesn't have to be very strict.

    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: PyTree,
        options: dict[str, Any],
    ) -> _IterativeDualState:
        # TODO(raderj): support operator_inv.
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
        if operator is None:
            assert False
        return _IterativeDualState(descent_state.y, descent_state.fn, vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _IterativeDualState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
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
        newton_step = (-ω(newton_soln.value)).ω
        newton_result = RESULTS.promote(newton_soln.result)

        def comparison_fn(
            lambda_i: Scalar,
            args: PyTree,
        ):
            step, _ = direct_dual(lambda_i)
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
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )
