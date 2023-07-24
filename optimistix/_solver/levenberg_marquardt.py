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
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar, ScalarLike

from .._custom_types import Aux, NoAuxFn, Out, Y
from .._misc import max_norm, two_norm
from .._root_find import AbstractRootFinder, root_find
from .._search import (
    AbstractDescent,
    AbstractSearch,
    damped_newton_step,
    DerivativeInfo,
    newton_step,
)
from .._solution import RESULTS
from .gauss_newton import AbstractGaussNewton
from .newton_chord import Newton
from .trust_region import ClassicalTrustRegion


class _Damped(eqx.Module):
    operator: lx.AbstractLinearOperator
    damping: Float[Array, " "]

    def __call__(self, y: PyTree[Array]):
        residual = self.operator.mv(y)
        damped = jtu.tree_map(lambda yi: jnp.sqrt(self.damping) * yi, y)
        return residual, damped


class DampedNewtonDescent(AbstractDescent[Y, Out, None]):
    """The damped Newton (Levenberg--Marquardt) descent.

    That is: gradient descent is about moving in the direction of `-grad`.
    (Quasi-)Newton descent is about moving in the direction of `-Hess^{-1} grad`. Damped
    Newton interpolates between these two regimes, by moving in the direction of
    `-(Hess + λI)^{-1} grad`.

    The value λ is often referred to as a the "Levenberg--Marquardt" parameter, and in
    version is handled directly, as λ = 1/step_size. Larger step sizes correspond to
    Newton directions; smaller step sizes correspond to gradient directions. (And
    additionally also reduces the step size, hence the term "damping".) This is because
    a line search expects the step to be smaller as the step size decreases.
    """

    # Will probably resolve to either Cholesky (for minimisation problems) or
    # QR (for least-squares problems).
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def optim_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        return None

    def search_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: None,
        deriv_info: DerivativeInfo,
    ) -> None:
        return None

    def descend(
        self,
        step_size: Scalar,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: None,
        deriv_info: DerivativeInfo,
    ) -> tuple[Y, RESULTS, None]:
        del fn, y, args, f, state
        sol_value, result = damped_newton_step(
            step_size, deriv_info, self.linear_solver
        )
        y_diff = (-(sol_value**ω)).ω
        return y_diff, result, None


DampedNewtonDescent.__init__.__doc__ = """**Arguments:**

- `linear_solver`: The linear solver used to compute the Newton step.
"""


_IndirectDampedNewtonDescentState: TypeAlias = tuple[Scalar, Y, Scalar, RESULTS]


class IndirectDampedNewtonDescent(
    AbstractDescent[Y, Out, _IndirectDampedNewtonDescentState]
):
    """The indirect damped Newton (Levenberg--Marquardt) trust-region descent.

    If the above line just looks like technical word soup, here's what's going on:

    Gradient descent is about moving in the direction of `-grad`. (Quasi-)Newton descent
    is about moving in the direction of `-Hess^{-1} grad`. Damped Newton interpolates
    between these two regimes, by moving in the direction of
    `-(Hess + λI)^{-1} grad`.

    This can be derived as the dual problem of a trust region method, see Conn, Gould,
    Toint: "Trust-Region Methods" section 7.3. λ is interpreted as a Lagrange
    multiplier. This involves solving a one-dimensional root-finding problem for λ at
    each descent.
    """

    lambda_0: ScalarLike = 1.0
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)
    # Default tol for `root_finder` because the tol doesn't have to be very strict.
    root_finder: AbstractRootFinder = Newton(rtol=1e-2, atol=1e-2)
    trust_region_norm: Callable[[PyTree], Scalar] = two_norm

    def optim_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> _IndirectDampedNewtonDescentState:
        # Dummy values
        return jnp.array(self.lambda_0), y, jnp.array(0.0), RESULTS.successful

    def search_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: _IndirectDampedNewtonDescentState,
        deriv_info: DerivativeInfo,
    ) -> _IndirectDampedNewtonDescentState:
        lambda_0, *_ = state
        newton_sol, newton_result = newton_step(deriv_info, self.linear_solver)
        newton_norm = self.trust_region_norm(newton_sol)
        return lambda_0, newton_sol, newton_norm, newton_result

    def descend(
        self,
        step_size: Scalar,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: _IndirectDampedNewtonDescentState,
        deriv_info: DerivativeInfo,
    ) -> tuple[Y, RESULTS, _IndirectDampedNewtonDescentState]:
        lambda_0, newton_sol, newton_norm, newton_result = state
        # For trust-region methods like `IndirectDampedNewtonDescent`, the trust-region
        # size directly controls how large a step we take. This is actually somewhat
        # annoying, as a trust region algorithm has no understanding of the scale of a
        # problem unless initialised with a complex initialisation procedure.
        #
        # A simple, heuristic way around this is to scale the trust region `step_size`
        # so that a `step_size` of `1` corresponds to the full length of the Newton
        # step (anything greater than `1` will still accept the Newton step.)
        scaled_step_size = newton_norm * step_size

        def comparison_fn(lambda_i: Scalar, args: PyTree):
            step, _ = damped_newton_step(1 / lambda_i, deriv_info, self.linear_solver)
            return self.trust_region_norm(step) - scaled_step_size

        def reject_newton():
            lambda_out = root_find(
                fn=comparison_fn,
                has_aux=False,
                solver=self.root_finder,
                y0=lambda_0,
                args=args,
                options=dict(lower=1e-5),
                max_steps=32,
                throw=False,
            ).value
            y_diff, result = damped_newton_step(
                1 / lambda_out, deriv_info, self.linear_solver
            )
            return lambda_out, y_diff, result

        def accept_newton():
            return lambda_0, newton_sol, newton_result

        # Only do a root-find if we have a small step size, and our Newton step was
        # successful.
        do_root_solve = (newton_result == RESULTS.successful) & (step_size < 1)
        lambda_out, neg_y_diff, new_result = lax.cond(
            do_root_solve, reject_newton, accept_newton
        )
        new_state = (lambda_out, newton_sol, newton_norm, newton_result)
        return (-(neg_y_diff**ω)).ω, new_result, new_state


IndirectDampedNewtonDescent.__init__.__doc__ = """**Arguments:**    

- `lambda_0`: The initial value of the Levenberg--Marquardt parameter used in the root-
    find to hit the trust-region radius. If `IndirectDampedNewtonDescent` is failing,
    this value may need to be increased.
- `linear_solver`: The linear solver used to compute the Newton step.
- `root_finder`: The root finder used to find the Levenberg--Marquardt parameter which
    hits the trust-region radius.
- `trust_region_norm`: The norm used to determine the trust-region shape.
"""


class LevenbergMarquardt(AbstractGaussNewton[Y, Out, Aux]):
    """The Levenberg--Marquardt method.

    This is a classical solver for nonlinear least squares, which works by regularising
    [`optimistix.GaussNewton`][] with a damping factor. This serves to (a) interpolate
    between Gauss--Newton and steepest descent, and (b) limit step size to a local
    region around the current point.

    This is a good algorithm for many least squares problems.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: AbstractDescent
    search: AbstractSearch

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = DampedNewtonDescent()
        self.search = ClassicalTrustRegion()


LevenbergMarquardt.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
"""


class IndirectLevenbergMarquardt(AbstractGaussNewton[Y, Out, Aux]):
    """The Levenberg--Marquardt method as a true trust-region method.

    This is a variant of [`optimistix.LevenbergMarquardt`][]. The other algorithm works
    by updating the damping factor directly -- this version instead updates a trust
    region, and then fits the damping factor to the size of the trust region.

    Generally speaking [`optimistix.LevenbergMarquardt`][] is preferred, as it performs
    nearly the same algorithm, without the computational overhead of an extra (scalar)
    nonlinear solve.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: AbstractDescent
    search: AbstractSearch

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        lambda_0: ScalarLike = 1.0,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False),
        root_finder: AbstractRootFinder = Newton(rtol=0.01, atol=0.01),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = IndirectDampedNewtonDescent(
            lambda_0=lambda_0,
            linear_solver=linear_solver,
            root_finder=root_finder,
        )
        self.search = ClassicalTrustRegion()


IndirectLevenbergMarquardt.__init__.__doc__ = """**Arguments:**
    
- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `lambda_0`: The initial value of the Levenberg--Marquardt parameter used in the root-
    find to hit the trust-region radius. If `IndirectLevenbergMarquardt` is failing,
    this value may need to be increased.
- `linear_solver`: The linear solver used to compute the Newton step.
- `root_finder`: The root finder used to find the Levenberg--Marquardt parameter which
    hits the trust-region radius.
"""
