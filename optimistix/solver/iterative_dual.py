import functools as ft
from typing import Callable, ClassVar, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, Float

from ..line_search import AbstractDescent, AbstractProxyDescent
from ..linear_operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    JacobianLinearOperator,
    linearise,
    PyTreeLinearOperator,
)
from ..linear_solve import AbstractLinearSolver, AutoLinearSolver, linear_solve
from ..linear_tags import positive_semidefinite_tag
from ..misc import NoneAux
from ..root_find import AbstractRootFinder, root_find, RootFindProblem
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
    damping: Float[ArrayLike, " "]

    def __call__(self, y, args):
        damping = jnp.sqrt(self.damping)
        f, aux = self.fn(y, args)
        damped = jtu.tree_map(lambda yi: damping * yi, y)
        return (f, damped), aux


class DirectIterativeDual(AbstractDescent):
    gauss_newton: bool
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=False)
    modify_jac: Callable[[JacobianLinearOperator], AbstractLinearOperator] = linearise
    computes_operator: bool = True
    computes_vector: ClassVar[bool] = False
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = True

    def __call__(
        self, delta, delta_args, problem, y, args, state, options, vector, operator
    ):
        # TODO(raderj): come up with a better way of handling this! We generally avoid
        # asking the descent to compute the operator directly and would rather be in the
        # solver level. However, this is definitely behavior that should be in the
        # descent. If we have a Gauss Newton method than the Jac will be passed in and
        # we shouldn't be recomputing it from the problem function, we should just
        # append on the sqrt(lambda) I. Same for non Gauss-Newton. How do we create
        # this operator?
        if self.gauss_newton:
            if isinstance(problem.fn, NoneAux):
                fn = NoneAux(problem.fn.fn.residual_fn)
            else:
                fn = problem.fn.residual_fn
            operator = JacobianLinearOperator(
                _Damped(fn, delta),
                y,
                args,
                _has_aux=True,
                # tags=positive_semidefinite_tag,
            )
            vector = (vector, ω(y).call(jnp.zeros_like).ω)
        else:
            # TODO(raderj): handle this case via block operators
            operator = operator + delta * IdentityLinearOperator(
                jax.eval_shape(lambda: operator)
            )
            operator_mat = operator.as_matrix()
            eye = jnp.eye(operator_mat.shape[0])
            operator = PyTreeLinearOperator(
                operator_mat + delta * eye,
                positive_semidefinite_tag,
                jax.eval_shape(lambda: vector),
            )
        operator = self.modify_jac(operator)

        eqxi.error_if(
            delta, delta < 1e-10, "The dual (LM) parameter must be >1e-10 to ensure psd"
        )
        linear_soln = linear_solve(operator, vector, self.solver)
        descent_dir = (-ω(linear_soln.value)).ω
        return descent_dir, linear_soln.aux


class IndirectIterativeDual(AbstractProxyDescent):

    #
    # Indirect iterative dual finds the `lambda` to match the
    # trust region radius by applying Newton's root finding method to
    # `phi.alt = 1/||p(lambda)|| - 1/delta`
    # As reccomended by [Hebden 1973 p. 8] and later [Nocedal Wright].
    # We could also use
    # `phi(lambda) = ||p(lambda)|| - delta`
    # but found it less numerically stable
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
    # NOTE/WARNING: this is not using the more efficient QR method.
    #
    gauss_newton: bool
    lambda_0: Float[ArrayLike, " "]
    root_finder: Optional[AbstractRootFinder]
    solver: AbstractLinearSolver
    tr_reg: PyTreeLinearOperator
    norm: Callable
    modify_jac: Callable[[JacobianLinearOperator], AbstractLinearOperator]
    computes_vector: ClassVar[bool] = False
    computes_operator: bool
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = True

    def __init__(
        self,
        gauss_newton: bool,
        lambda_0: Float[ArrayLike, " "],
        root_finder: Optional[AbstractRootFinder] = None,
        solver=AutoLinearSolver(well_posed=False),
        tr_reg: Optional[PyTreeLinearOperator] = None,
        norm: Callable = jnp.linalg.norm,
        modify_jac: Callable[
            [JacobianLinearOperator], AbstractLinearOperator
        ] = linearise,
    ):
        self.lambda_0 = lambda_0
        self.norm = norm
        self.tr_reg = tr_reg
        self.modify_jac = modify_jac
        self.solver = solver

        # it a gauss_newton method, then the model will compute the Jacobian
        self.gauss_newton = gauss_newton
        self.computes_operator = gauss_newton
        # WARNING: the intended behavior is that the user would pass Newton(atol, rtol)
        # themselves if they want a specific atol/rtol, but this may be a poor design
        # choice and we can just ask for atol, rtol.
        if root_finder is None:
            # being super precise is not very important in the Newton step
            self.root_finder = Newton(rtol=1e-1, atol=1e-2, lower=0.0)
        else:
            self.root_finder = root_finder

    def __call__(
        self, delta, delta_args, problem, y, args, state, options, vector, operator
    ):
        direct_dual = DirectIterativeDual(
            self.gauss_newton, self.solver, self.modify_jac
        )
        direct_dual = eqxi.Partial(
            direct_dual,
            delta_args=None,
            problem=problem,
            y=y,
            state=state,
            args=args,
            options=options,
            vector=vector,
            operator=operator,
        )

        tr_reg = self.tr_reg
        if tr_reg is None:
            tr_reg = IdentityLinearOperator(jax.eval_shape(lambda: y))

        linear_soln = linear_solve(operator, vector, self.solver)
        newton_step = (-ω(linear_soln.value)).ω
        newton_aux = linear_soln.aux

        args = (self, delta, newton_step, problem, direct_dual, tr_reg, newton_aux)
        dynamic_args, static_args = eqx.partition(args, eqx.is_inexact_array)

        def accept_newton(dynamic_args):
            args = eqx.combine(dynamic_args, static_args)
            self, delta, newton_step, problem, direct_dual, tr_reg, newton_aux = args
            return newton_step, newton_aux

        def reject_newton(dynamic_args):
            args = eqx.combine(dynamic_args, static_args)
            self, delta, newton_step, problem, direct_dual, tr_reg, newton_aux = args
            comparison_fn = ft.partial(
                self.comparison_fn, delta=delta, direct_dual=direct_dual, tr_reg=tr_reg
            )
            rf_problem = RootFindProblem(fn=comparison_fn, has_aux=False)
            lambda_out = root_find(
                rf_problem, self.root_finder, self.lambda_0, args, options
            ).value
            return direct_dual(lambda_out)

        ravel_newton, _ = jax.flatten_util.ravel_pytree(tr_reg.mv(newton_step))
        newton_norm = self.norm(ravel_newton)

        jax.debug.print("Newton norm: {}", newton_norm)
        return lax.cond(newton_norm < delta, accept_newton, reject_newton, dynamic_args)

    def comparison_fn(self, lambda_i, lambda_i_args, delta, direct_dual, tr_reg):
        jax.debug.print("lambda_i: {}", lambda_i)
        (step, _) = direct_dual(lambda_i)
        (step_test, _) = direct_dual(lambda_i + 1)
        # TODO(raderj): should just be self.norm! But we need to assure that
        # self.norm acts on pytrees, in which case this tree_reduce can be
        # removed as well
        ravel_step, _ = jax.flatten_util.ravel_pytree(step)
        step_norm = self.norm(ravel_step)
        # step_norm = jtu.tree_reduce(
        #     lambda x,y: x + y, ω(step).call(lambda x: jnp.linalg.norm(x)**2).ω
        # )

        jax.debug.print("Delta: {}", delta)
        jax.debug.print("Step_norm: {}", step_norm)

        return 1 / delta - 1 / step_norm

    def predicted_reduction(self, descent_dir, args, state, options, vector, operator):
        model_0 = self.norm(vector) ** 2
        model_p = self.norm((ω(operator.mv(descent_dir)) - ω(vector)).ω) ** 2
        return 0.5 * (model_0 - model_p)
