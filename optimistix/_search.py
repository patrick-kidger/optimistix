"""We use notions of "searches" and "descents" to generalise line searches, trust
regions, and learning rates.

For example, the classical gradient descent algorithm is given by setting the search
to be `optimistix.LearningRate`, and setting the descent to be
`optimistix.SteepestDescent`.

As another example, Levenberg--Marquardt is given by setting the search to be
`optimistix.ClassicalTrustRegion`, and setting the descent to be
`optimistix.DampedNewtonDescent`.

As a third example, BFGS is given by setting the search to be
`optimistix.BacktrackingArmijo`, and setting the descent to be
`optimistix.NewtonDescent`.

This gives us a flexible way to mix-and-match ideas across different optimisers.

Right now, these are used exclusively for minimisers and least-squares optimisers. (As
the latter are a special case of the former, with `to_minimise = 0.5 * residuals^2`)

For an in-depth discussion of how these pieces fit together, see the documentation at
`docs/api/searches/introduction.md`.
"""

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
from typing import Any, ClassVar, Generic, Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from ._custom_types import (
    DescentState,
    NoAuxFn,
    Out,
    SearchState,
    Y,
)
from ._misc import tree_full_like
from ._solution import RESULTS


class DerivativeInfo(eqx.Module):
    """Different solvers (BFGS, Levenberg--Marquardt, ...) evaluate different quantities
    to do with the derivative of the objective function. This enumeration-ish object
    captures the different variants.

    Available variants are
    `optimistix.DerivativeInfo.{Grad, GradHessian, GradHessianInv, Jac}`.

    Each [`optimsitix.AbstractSearch`][] and [`optimistix.AbstractDescent`][] may only
    accept some variants. For example, [`optimistix.NewtonDescent`][] requires a
    Hessian approximation, which [`optimistix.NonlinearCG`][] does not attempt to
    estimate.
    """

    Grad: ClassVar[Type["_Grad"]]
    GradHessian: ClassVar[Type["_GradHessian"]]
    GradHessianInv: ClassVar[Type["_GradHessianInv"]]
    ResidualJac: ClassVar[Type["_ResidualJac"]]


class _Grad(DerivativeInfo, Generic[Y]):
    """Has a `.grad` attribute describing `d(fn)/dy`. Used with first-order solvers for
    minimisation problems. (E.g. gradient descent; nonlinear CG.)
    """

    grad: Y


class _GradHessian(DerivativeInfo, Generic[Y]):
    """Has a `.grad` attribute as with `optximistix.DerivativeInfo.Grad`, and also a
    `.hessian` attribute describing (an approximation to) the Hessian of `fn` at `y`.
    Used with quasi-Newton minimisation algorithms, like BFGS.
    """

    grad: Y
    hessian: lx.AbstractLinearOperator


class _GradHessianInv(DerivativeInfo, Generic[Y]):
    """As `optimistix.DerivativeInfo.GradHessian`, but records the (approximate)
    inverse-Hessian instead. Has `.grad` and `.hessian_inv` attributes.
    """

    grad: Y
    hessian_inv: lx.AbstractLinearOperator


class _ResidualJac(DerivativeInfo, Generic[Y, Out]):
    """Records the Jacobian `d(fn)/dy` as a linear operator. Used for least squares
    problems, for which `fn` returns residuals. Has `.residual` and `.jac` and `.grad`
    attributes, where `residual = fn(y, args)`, `jac = d(fn)/dy` and
    `grad = jac^T residual`.

    Takes just `residual` and `jac` as `__init__`-time arguments, from which `grad` is
    computed.
    """

    residual: Out
    jac: lx.AbstractLinearOperator
    grad: Y

    def __init__(self, residual: Out, jac: lx.AbstractLinearOperator):
        self.residual = residual
        self.jac = jac
        # The gradient is used ubiquitously, so compute it once here, so that it can be
        # used without recomputation in both the descent and search.
        self.grad = jac.transpose().mv(residual)


_Grad.__name__ = "Grad"
_Grad.__qualname__ = "DerivativeInfo.Grad"
_GradHessian.__name__ = "GradHessian"
_GradHessian.__qualname__ = "DerivativeInfo.GradHessian"
_GradHessianInv.__name__ = "GradHessianInv"
_GradHessianInv.__qualname__ = "DerivativeInfo.GradHessianInv"
_ResidualJac.__name__ = "Jac"
_ResidualJac.__qualname__ = "DerivativeInfo.Jac"
DerivativeInfo.Grad = _Grad
DerivativeInfo.GradHessian = _GradHessian
DerivativeInfo.GradHessianInv = _GradHessianInv
DerivativeInfo.ResidualJac = _ResidualJac


class AbstractDescent(eqx.Module, Generic[Y, Out, DescentState]):
    """The abstract base class for descents. A descent consumes a scalar (e.g. a step
    size), and returns the `diff` to take at point `y`, so that `y + diff` is the next
    iterate in a nonlinear optimisation problem.

    See [this documentation](./introduction.md) for more information.
    """

    @abc.abstractmethod
    def optim_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> DescentState:
        """Is called just once, at the very start of the entire optimisation problem.
        This is used to set up the evolving state for the descent.

        In practice, this is typically called from within (an implementation of)
        [`optimistix.AbstractSearch.init`][].

        **Arguments:**

        - `fn`: the problem function to optimise. Typically either a minimisation
            problem or a least-squares problem.
        - `y`: The initial guess for the optimisation problem.
        - `args`: as passed to the overall optimisation problem. Will be forwarded on to
            `fn`.
        - `f_struct`: a PyTree of `jax.ShapeDtypeStruct`s describing the structure of
            the output of `fn`.

        **Returns:**

        The initial state, prior to doing any descents or searches.
        """

    @abc.abstractmethod
    def search_init(
        self,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: DescentState,
        deriv_info: DerivativeInfo,
    ) -> DescentState:
        """Is called at the start of each search.

        This is passed all the same information as `descend`. This method can be used to
        precompute information that does not depend on the step size.

        In practice, this is typically called at the start of (an implementation of)
        [`optimistix.AbstractSearch.search`][].

        **Arguments:**

        - `fn`: the problem function to optimise. Typically either a minimisation
            problem or a least-squares problem.
        - `y`: the value of the current iterate.
        - `args`: as passed to the overall optimisation problem. Will be forwarded on to
            `fn`.
        - `f`: the value of `fn(y0, args)`. (Whilst we could just evaluate this inside
            `search`, it is common for this to already be available, so as a performance
            optimisation we accept it as an argument instead.)
        - `state`: the evolving state of the repeated descents.
        - `deriv_info`: an [`optimistix.DerivativeInfo`][] describing the derivative
            information available at this point.

        **Returns:**

        The updated descent state.
        """

    @abc.abstractmethod
    def descend(
        self,
        step_size: Scalar,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: DescentState,
        deriv_info: DerivativeInfo,
    ) -> tuple[Y, RESULTS, DescentState]:
        """Computes the descent direction.

        In practice, this is typically called (potentially multiple times) inside (an
        implementation of) [`optimistix.AbstractSearch.search`][].

        **Arguments:**

        - `step_size`: a non-negative scalar describing the step size to take.
        - `fn`: the problem function to optimise. Typically either a minimisation
            problem or a least-squares problem.
        - `y`: the value of the current iterate.
        - `args`: as passed to the overall optimisation problem. Will be forwarded on to
            `fn`.
        - `f`: the value of `fn(y0, args)`. (Whilst we could just evaluate this inside
            `search`, it is common for this to already be available, so as a performance
            optimisation we accept it as an argument instead.)
        - `state`: the evolving state of the repeated descents.
        - `deriv_info`: an [`optimistix.DerivativeInfo`][] describing the derivative
            information available at this point.

        **Returns:**

        A 3-tuple of `(y_diff, result, new_state)`, describing the delta to apply in
        y-space, any error describing whether the state was successful, and the updated
        state for the next descent call.
        """


class AbstractSearch(eqx.Module, Generic[Y, Out, SearchState]):
    """The abstract base class for all searches. (Which are our generalisation of
    line searches, trust regions, and learning rates.)

    See [this documentation](./introduction.md) for more information.
    """

    @abc.abstractmethod
    def init(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Scalar],
        y: Y,
        args: PyTree,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> SearchState:
        """Is called just once, at the very start of the entire optimisation problem.
        This is used to set up the evolving state for the sequence of searches.

        For example, trust region methods pass the last trust region radius between
        invocations.

        **Arguments:**

        - `descent`: an [`optimistix.AbstractDescent`][] object describing how to move
            around in y-space. (E.g. steepest descent, dogleg descent, ...)
        - `fn`: the problem function to optimise. Typically either a minimisation
            problem or a least-squares problem.
        - `y`: The initial guess for the optimisation problem.
        - `args`: as passed to the overall optimisation problem. Will be forwarded on to
            `fn`.
        - `f_struct`: a PyTree of `jax.ShapeDtypeStruct`s describing the structure of
            the output of `fn`.

        **Returns:**

        The initial state for the sequence of searches.
        """

    @abc.abstractmethod
    def search(
        self,
        descent: AbstractDescent,
        fn: NoAuxFn[Y, Out],
        y: Y,
        args: PyTree[Any],
        f: Out,
        state: SearchState,
        deriv_info: DerivativeInfo,
        max_steps: Optional[int],
    ) -> tuple[Y, Bool[Array, ""], RESULTS, SearchState]:
        """Performs a search.

        For example, this may perform a line search. (The entire line search, not just
        a single step in the line search.)

        **Arguments:**

        - `descent`: an [`optimistix.AbstractDescent`][] object describing how to move
            around in y-space. (E.g. steepest descent, dogleg descent, ...)
        - `fn`: the problem function to optimise. Typically either a minimisation
            problem or a least-squares problem.
        - `y`: the value of the current iterate.
        - `args`: as passed to the overall optimisation problem. Will be forwarded on to
            `fn`.
        - `f`: the value of `fn(y0, args)`. (Whilst we could just evaluate this inside
            `search`, it is common for this to already be available, so as a performance
            optimisation we accept it as an argument instead.)
        - `state`: the evolving state of the repeated searches.
        - `deriv_info`: an [`optimistix.DerivativeInfo`][] describing the derivative
            information available at this point.
        - `max_steps`: for line searches, the maximum number of steps allowed in the
            line search.

        **Returns:**

        A 3-tuple of `(y_diff, result, new_state)`, describing the delta to apply in
        y-space (so that the location of the next iterate should be `y + y_diff`), any
        errors encountered during the search, and the next search state.
        """


def newton_step(
    deriv_info: DerivativeInfo, linear_solver: lx.AbstractLinearSolver
) -> tuple[PyTree[Array], RESULTS]:
    """Compute a Newton step.

    For a minimisation problem, this means computing `Hess^{-1} grad`.

    For a least-squares problem, we convert to a minimisation problem via
    `0.5*residuals^2`, which then implies `Hess^{-1} ~ J^T J` (Gauss--Newton
    approximation) and `grad = J^T residuals`.

    Thus `Hess^{-1} grad ~ (J^T J)^{-1} J^T residuals`.   [Equation A]

    Now if `J` is well-posed then this equals `J^{-1} residuals`, which is exactly what
    we compute here.

    And if `J` is ill-posed then [Equation A] is just the normal equations, which should
    almost never be treated directly! (Squares the condition number blahblahblah.) The
    solution of the normal equations matches the pseudoinverse solution `J^{dagger}`
    residuals, which is what we get using an ill-posed-capable linear solver (typically
    QR). So we solve the same linear system as in the well-posed case, we just need to
    set a different linear solver. (Which happens with
    `linear_solver=lx.AutoLinearSolver(well_posed=None)`, which is the recommended
    value.)
    """
    if isinstance(deriv_info, DerivativeInfo.Grad):
        raise ValueError(
            "Cannot use a Newton descent with a minimiser that only evaluates the "
            "gradient."
        )
    elif isinstance(deriv_info, DerivativeInfo.GradHessianInv):
        newton = deriv_info.hessian_inv.mv(deriv_info.grad)
        result = RESULTS.successful
    else:
        if isinstance(deriv_info, DerivativeInfo.GradHessian):
            operator = deriv_info.hessian
            vector = deriv_info.grad
        elif isinstance(deriv_info, DerivativeInfo.ResidualJac):
            operator = deriv_info.jac
            vector = deriv_info.residual
        else:
            assert False
        out = lx.linear_solve(operator, vector, linear_solver)
        newton = out.value
        result = RESULTS.promote(out.result)
    return newton, result


class _Damped(eqx.Module):
    operator: lx.AbstractLinearOperator
    damping: Float[Array, " "]

    def __call__(self, y: PyTree[Array]):
        residual = self.operator.mv(y)
        damped = jtu.tree_map(lambda yi: jnp.sqrt(self.damping) * yi, y)
        return residual, damped


def damped_newton_step(
    step_size: Scalar,
    deriv_info: DerivativeInfo,
    linear_solver: lx.AbstractLinearSolver,
) -> tuple[PyTree[Array], RESULTS]:
    """Compute a damped Newton step.

    For a minimisation problem, this means solving `(Hess + λI)^{-1} grad`.

    In the (nonlinear) least-squares case, for which the minimisation objective
    is given by `0.5*residual^2`, then we know that `grad=J^T residual`, and we make
    the Gauss--Newton approximation `Hess ~ J^T J`. This reduces the above to
    solving the (linear) least-squares problem
    ```
    [    J   ] [diff]  =  [residual]
    [sqrt(λ)I]         =  [   0    ]
    ```
    This can be seen by observing that the normal equations for the this linear
    least-squares problem is the original linear problem we wanted to solve.
    """

    pred = step_size > jnp.finfo(step_size.dtype).eps
    safe_step_size = jnp.where(pred, step_size, 1)
    lm_param = jnp.where(pred, 1 / safe_step_size, jnp.finfo(step_size).max)
    if isinstance(deriv_info, (DerivativeInfo.Grad, DerivativeInfo.GradHessianInv)):
        raise ValueError(
            "Damped newton descent cannot be used with a solver that does not "
            "provide (approximate) Hessian information."
        )
    elif isinstance(deriv_info, DerivativeInfo.GradHessian):
        operator = deriv_info.hessian + lm_param * lx.IdentityLinearOperator(
            deriv_info.hessian.in_structure()
        )
        vector = deriv_info.grad
        if lx.is_positive_semidefinite(deriv_info.hessian):
            operator = lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
    elif isinstance(deriv_info, DerivativeInfo.ResidualJac):
        y_structure = deriv_info.jac.in_structure()
        operator = lx.FunctionLinearOperator(
            _Damped(deriv_info.jac, lm_param), y_structure
        )
        vector = (deriv_info.residual, tree_full_like(y_structure, 0))
    else:
        assert False
    linear_sol = lx.linear_solve(operator, vector, linear_solver, throw=False)
    return linear_sol.value, RESULTS.promote(linear_sol.result)
