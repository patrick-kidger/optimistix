import functools as ft
from typing import Callable, ClassVar, Optional

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..custom_types import sentinel
from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractTRModel
from ..linear_operator import (
    AbstractLinearOperator,
    JacobianLinearOperator,
    linearise,
    MatrixLinearOperator,
)
from ..linear_solve import AbstractLinearSolver, AutoLinearSolver, linear_solve
from ..linear_tags import positive_semidefinite_tag
from ..root_find import AbstractRootFinder, root_find
from ..solver import Newton, QR


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
# trustregion radius), `I` is the identity, and `r` is the vector of residuals.
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
# The indirect approach is very interpretable in the classical trust region sense.
# Note however, if `B` is the quasi-Newton Hessian approximation and `g` the
# gradient, that `||p(lambda)||` is dependent upon `B`. Specifically, decompose
# `B = QLQ^T` via the spectral decomposition with eigenvectors $q_i$ and corresponding
# eigenvalues $l_i$. Then
# ```
# ||p(lambda)||^2 = sum(((q_j^T g)^2)/(l_1 + lambda)^2)
# ```
# The consequence of this is that the relationship between lambda and the trust region
# radius changes each iteration as `B` is updated. For this reason, I (raderj) tend
# towards the more interpretable indirect approach. This is what Moré used in their
# classical implementation of the algorithm as well (see Moré, "The Levenberg-Marquardt
# Algorithm: Implementation and Theory.")
#

#
# TODO(raderj): handle the case where we pass in a nonsingular L into the trust region.
#


class _Damped(eqx.Module):
    fn: Callable
    damping: float

    def __call__(self, y, args):
        damping = jnp.sqrt(self.damping)
        f, aux = self.fn(y, args)
        damped = jtu.tree_map(lambda yi: damping * yi, y)
        return (f, damped), aux


class DirectIterativeDual(AbstractTRModel):
    gauss_newton: bool
    solver: AbstractLinearSolver
    modify_jac: Callable[[JacobianLinearOperator], AbstractLinearOperator]
    computes_operator: bool = True
    computes_vector: ClassVar[bool] = False

    def __init__(
        self,
        gauss_newton: bool,
        solver: AbstractLinearSolver = AutoLinearSolver,
        modify_jac=linearise,
    ):
        # do I need to do this?
        self.solver, self.modify_jac = solver, modify_jac
        self.gauss_newton = self.computes_operator = gauss_newton

    def descent_dir(self, delta, state):
        def raise_error():
            raise ValueError(
                "The dual (Levenberg-Marquardt) parameter must be larger than \
                10^-6 to make sure things are psd."
            )

        def no_error():
            pass

        if self.gauss_newton:
            operator = JacobianLinearOperator(
                _Damped(state.problem.fn, delta),
                state.y,
                state.args,
                _has_aux=True,
                tags=positive_semidefinite_tag,
            )
            operator = self.modify_jac(operator)
        else:
            operator_mat = state.operator.as_matrix()
            eye = jnp.eye(operator_mat.shape[0])
            operator = MatrixLinearOperator(
                operator_mat + delta * eye, positive_semidefinite_tag
            )

        lax.cond(delta < 1e-6, raise_error, no_error)
        return -self.solver(operator, state.vector)


class IndirectIterativeDual(AbstractTRModel):

    #
    # Indirect iterative dual finds the `lambda` to match the
    # trust region radius by applying Newton's root finding method to
    # `phi(lambda) = ||p(lambda)|| - delta`
    # Note that [Hebden 1973 p. 8] and later [Nocedal Wright] reccomend against
    # this choice of phi! They instead propose reparamterising
    # `phi.alt = 1/||p(lambda)|| - 1/delta`
    #
    # Moré found a clever way to compute `phi = -||q||^2/||p(lambda)||` where `q` is
    # defined as: `q = R^(-1) p`, for `R` as in the QR decomposition of
    # `(B + lambda I)`. This can similarly be applied to the phi in Hebden with
    # `phi' = ||q||^2/||p(lambda)||^(3/2)`
    #
    # Note, however, that we are going to totally ignore this neat trick (gasp!)
    # We view modularity of the phi function as of greater benefit than
    # the explicit form of `phi'`, which we compute with automatic differentiation
    # in the Newton solve.
    #
    # TODO(raderj): write a solver in root_finder which specifically assumes iterative
    # dual so we can use the trick (or at least see if it's worth doing.)
    #
    gauss_newton: bool
    atol: float
    rtol: float
    lambda_0: Float[ArrayLike, " "]
    computes_operator: bool
    tr_matrix: Optional[PyTree[ArrayLike]] = sentinel
    computes_vector: ClassVar[bool] = False
    norm: Callable = jnp.linalg.norm
    root_finder: AbstractRootFinder = Newton
    modify_jac: Callable[[JacobianLinearOperator], AbstractLinearOperator] = linearise

    def __init__(
        self,
        gauss_newton: bool,
        atol: float,
        rtol: float,
        lambda_0: Float[ArrayLike, " "],
        tr_matrix=sentinel,
        norm=jnp.linalg.norm,
        modify_jac=linearise,
    ):
        self.atol, self.rtol, self.lambda_0, self.norm = atol, rtol, lambda_0, norm
        self.tr_matrix, self.modify_jac = tr_matrix, modify_jac

        self.gauss_newton = self.computes_operator = gauss_newton

    def descent_dir(self, delta, state):
        # TODO(raderj): add support for a nonsingular matrix to solve the tr
        # subproblem `||Lp|| < delta`
        def accept_newton(self, delta, state, newton_step):
            return newton_step

        def reject_newton(self, delta, state, newton_step):
            direct_dual = DirectIterativeDual(self.gauss_newton, QR(), self.modify_jac)

            comparison_fn = ft.partial(
                self.comparison_fn, state=state, direct_dual=direct_dual
            )
            problem = AbstractIterativeProblem(fn=comparison_fn, has_aux=False)
            lambda_out = root_find(
                problem, Newton(), self.lambda_0, state.args, state.options
            )
            return lambda_out

        newton_step = -linear_solve(state.operator, state.vector).value

        args = (self, delta, state, newton_step)

        if self.tr_matrix != sentinel:
            newton_conditioned = ω(self.tr_matrix) @ ω(newton_step)
        else:
            newton_conditioned = ω(newton_step)

        return lax.cond(
            newton_conditioned.call(self.norm).ω < delta,
            accept_newton,
            reject_newton,
            args,
        )

        def comparison_fn(self, lambda_i, state, direct_dual):
            step = self.direct_dual.descent_dir(lambda_i, state).value
            return (ω(self.tr_matrix) @ ω(step)).call(self.norm).ω - lambda_i
