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

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar

from .._custom_types import Y
from .._descent import AbstractDescent
from .._misc import max_norm, tree_full_like, tree_where, two_norm
from .._root_find import AbstractRootFinder, root_find
from .._solution import RESULTS
from .gauss_newton import AbstractGaussNewton
from .newton_chord import Newton
from .trust_region import ClassicalTrustRegion


#
# NOTE: The descents `DirectIterativeDual` and `IndirectIterativeDual` are usually
# both referred to as Levenberg-Marquardt. However, Levenberg-Marquardt often
# refers specifically to the case where these descents are applied in the
# Gauss-Newton setting, ie. to solve a least-squares problem. However, both methods
# can be used more generally for nonlinear optimisation. For this reason, we
# refer to the descents by the more generic name "iterative dual."
#
# Iterative dual computes the descent `p` as the solution to: `(B + λ I) p = -g`.
# Here `B` is the approximate hessian, `λ` is the "Levenberg-Marquardt parameter,"
# `I` is the identity, and `g` is the gradient. In the Gauss-Newton setting,
# `B = J^T J` and `g = J^T r` where `J` is the Jacobian and `r` the residual vector.
#
# The equation `(B + λ I) p = -g` is derived as the dual problem of a trust region
# method (see Conn, Gould, Toint "Trust-Region Methods" section 7.3,) where
# `λ` is interpreted as a Lagrange multiplier. This allows iterative dual to
# be used as a trust-region method.
#
# This has lead to two approaches to the iterative dual method:
#
# 1. Use iterative dual as a trust-region method: set the trust region
# radius and find the Levenberg-Marquadt parameter `λ` which will approximately
# solve the trust region problem.
#
# 2. Set and update the Levenberg-Marquadt parameter `λ` directly.
#
# Respectively, this is the indirect and direct approach to iterative dual.
# The direct approach is common in practice, and can be interpreted as
# interpolating between quasi-Newton and gradient based approaches. The direct method
# is the method used in `LevenbergMarquardt`.
#
# In `DirectIterativeDual` we parameterise `λ = 1/step_size`, as
# a line search expects the step size to decrease as `step_size` decreases.
#
# NOTE: In the Gauss-Newton case, rather than solve
# `(J^T J + 1/step_size I) p = -J^T r`
# by computing`J^T J` and squaring the condition number of `J`, we can
# compute `[J, jnp.sqrt(1/step_size) I]^T p = [-r, 0]^T`.
# This is a common numerical approach.
#


class _Damped(eqx.Module):
    fn: Callable
    damping: Float[Array, " "]

    def __call__(self, y: PyTree[Array], args: PyTree):
        damping = jnp.sqrt(self.damping)
        residual, aux = self.fn(y, args)
        damped = jtu.tree_map(lambda yi: damping * yi, y)
        return (residual, damped), aux


class DirectIterativeDual(AbstractDescent[Y]):
    gauss_newton: bool

    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        vector = options["vector"]
        operator = options["operator"]
        lm_param = jnp.where(
            step_size > jnp.finfo(step_size.dtype).eps,
            1 / step_size,
            jnp.finfo(step_size).max,
        )
        if self.gauss_newton:
            # TODO(raderj): remove both of these options! Use block operators
            # to remove the call to `fn`. We can get the shape of `y` via
            # `operator.out_structure`.
            y = options["y"]
            fn = options["fn"]
            vector = (vector, ω(y).call(jnp.zeros_like).ω)
            operator = lx.JacobianLinearOperator(
                _Damped(fn, lm_param),
                y,
                args,
                _has_aux=True,
            )
        else:
            vector = vector
            operator = operator + lm_param * lx.IdentityLinearOperator(
                operator.in_structure()
            )
        linear_sol = lx.linear_solve(operator, vector, lx.QR(), throw=False)
        diff = (-linear_sol.value**ω).ω
        # TODO(raderj): Remove the following section,
        # it still computes `nan`s and is just a bandaid.
        # It seems that using `_Damped` with inf values
        # creates a `JacobianLinearOperator` with `nan`s, which leads to incorrect
        # linear solves. This won't be an issue when we switch to block matrix
        # operator but may need to raise an issue against Lineax.
        step_size_zero = step_size < jnp.finfo(step_size.dtype).eps
        diff = tree_where(step_size_zero, tree_full_like(diff, 0), diff)
        RESULTS.where(
            step_size_zero, RESULTS.successful, RESULTS.promote(linear_sol.result)
        )
        return diff, RESULTS.promote(linear_sol.result)


class IndirectIterativeDual(AbstractDescent[Y]):
    gauss_newton: bool
    lambda_0: float
    # Default tol for `root_finder` only because a user would override this entirely
    # with `FooRootFinder(rtol, atol, ...)` and the tol doesn't have to be very strict.
    root_finder: AbstractRootFinder = Newton(rtol=1e-2, atol=1e-2, lower=1e-5)
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)
    trust_region_norm: Callable = two_norm

    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        operator = options["operator"]
        vector = options["vector"]
        direct_dual = eqx.Partial(
            DirectIterativeDual(self.gauss_newton),
            args=args,
            options=options,
        )
        newton_soln = lx.linear_solve(
            operator,
            (-(vector**ω)).ω,
            self.linear_solver,
            throw=False,
        )
        newton_step = (-ω(newton_soln.value)).ω
        newton_norm = self.trust_region_norm(newton_step)
        newton_result = RESULTS.promote(newton_soln.result)

        #
        # For trust-region methods like `IndirectIterativeDual`, the trust-region
        # size directly controls how large a step we take. This is actually somewhat
        # annoying, as a trust region algorithm has no understanding of the scale of a
        # problem unless initialised with a complex initialisation procedure.
        #
        # A simple, heuristic way around this is to scale the trust region `step_size`
        # so that a `step_size` of `1` corresponds to the full length of the Newton
        # step (anything greater than `1` will still accept the Newton step.)
        #
        scaled_step_size = newton_norm * step_size

        def comparison_fn(lambda_i: Scalar, args: PyTree):
            step, _ = direct_dual(1 / lambda_i)
            return self.trust_region_norm(step) - scaled_step_size

        def accept_newton():
            return newton_step, newton_result

        def reject_newton():
            lambda_out = root_find(
                fn=comparison_fn,
                has_aux=False,
                solver=self.root_finder,
                y0=jnp.array(self.lambda_0),
                args=args,
                options={},
                max_steps=32,
                throw=False,
            ).value
            return direct_dual(1 / lambda_out)

        return lax.cond(step_size >= 1, accept_newton, reject_newton)


class LevenbergMarquardt(AbstractGaussNewton):
    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable = max_norm,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.line_search = ClassicalTrustRegion(
            DirectIterativeDual(gauss_newton=True), gauss_newton=True
        )


class IndirectLevenbergMarquardt(AbstractGaussNewton):
    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable = max_norm,
        lambda_0: float = 1.0,
        root_finder: AbstractRootFinder = Newton(rtol=0.01, atol=0.01, lower=1e-5),
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.line_search = ClassicalTrustRegion(
            IndirectIterativeDual(gauss_newton=True, lambda_0=lambda_0),
            gauss_newton=True,
        )
