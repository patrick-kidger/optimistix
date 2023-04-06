from typing import Callable, Optional

import jax.lax as lax
import jax.numpy as jnp
from equinox import ω
from jaxtyping import ArrayLike, Float

import optimistix as optx

from ..custom_types import sentinel
from ..line_search import AbstractQuasiNewtonTR


#
# WARNING: This is using optx and not just calling into the respective
# files, which is inconsistent with everywher else
#

#
# Levenberg-Marquardt is a method of solving for the descent direction
# given a trust region radius. It does this by solving the dual problem
# `(B + lambda I) p = r` for `p`, where `B` is the quasi-Newton matrix, lambda is the
# Levenberg-Marquardt parameter (the dual parameterisation of the trust region
# radius), `I` is the identity, and `r` is the vector of residuals.
#
# Levenberg-Marquardt is approached in one of two ways:
# 1. set the trust region radius and find the Levenberg-Marquadt parameter
# `lambda` which will give approximately solve the trust region problem. ie.
# solve the dual of the trust region problem.
#
# 2. set the Levenberg-Marquadt parameter `lambda` directly.
#
# Respectively, this is the indirect and direct approach to Levenberg-Marquardt.
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


class DirectLevenbergMarquardt(AbstractQuasiNewtonTR):
    tr_matrix: Optional[ArrayLike] = sentinel

    def descent_dir(self, delta, state):
        def raise_error():
            raise ValueError(
                "The Levenberg-Marquardt parameter must be larger than \
                10^-6 to make sure things are psd."
            )

        def no_error():
            pass

        lax.cond(delta < 1e-6, raise_error, no_error)
        hess_mat = state.hessian.as_matrix()
        eye = jnp.eye(hess_mat.shape[0])
        tikhonov = optx.MatrixLinearOperator(
            hess_mat + delta * eye, optx.positive_semidefinite_tag
        )
        return -optx.linear_solve(tikhonov, state.gradient)


class IndirectLevenbergMarquardt(AbstractQuasiNewtonTR):

    #
    # Indirect Levenberg-Marquardt finds the `lambda` to match the
    # trust region radius by applying Newton's root finding method to
    # `phi(lambda) = ||p(lambda)|| - delta`
    # Moré found a clever way to compute `phi = -||q||^2/||p(lambda)||` where `q` is
    # defined as: `q = R^(-1) p`, for `R` the Cholesky factor of `(B + lambda I)`.
    #
    # Note that Nocedal Wright actually recommends against using this phi!
    # They instead propose reparamterising `phi.alt = 1/||p(lambda)|| - 1/delta`
    # which leads to `phi.alt' = ||q||^2/||p(lambda)||^(3/2)`
    #
    # There are other variations of this as well (see Hebden 1973  p. 8) but we
    # stick with Moré's implementation as this is the most classical.
    #
    # There is an argument to just using a standard Newton root finding algorithm and
    # calling it a day. These algorithms were invented before automatic differentiation
    # became as ubiquitious as it is today. These may still be faster, but I would guess
    # the effect is marginal compared to the extra difficulty implementing of them.
    #
    # TODO(raderj): handle this properly :(, this is not set up to handle gauss-newton
    # methods at all, which is a big no-no.
    atol: float
    rtol: float
    lambda_0: Float[ArrayLike, " "]
    norm: Callable = jnp.linalg.norm

    def descent_dir(self, delta, state):
        # TODO(raderj): add support for a nonsingular matrix to solve the tr
        # subproblem `||Lp|| < delta`
        def accept_newton(self, delta, state, newton_step, hess, eye):
            return newton_step

        def reject_newton(self, delta, state, newton_step, hess, eye):
            l_0 = jnp.array(0.0)
            u_0 = ω(state.gradient).call(self.norm).ω / delta
            tikhonov = optx.MatrixLinearOperator(
                hess_mat + self.lambda_0 * eye, optx.positive_semidefinite_tag
            )
            init_step = -optx.linear_solve(tikhonov, state.gradient)

            # some stuff used in the termination check
            init_size = ω(init_step).call(self.norm).ω

            init = (l_0, u_0, self.lambda_0, state, init_step, hess, eye, init_size)

            def cond_fun(carry):
                _, _, _, _, step, _, _, init_size = carry
                return (
                    jnp.abs(delta - ω(step).call(self.norm).ω)
                    < self.atol + self.rtol * init_size
                )

            def body_fun(carry):
                lower_i, upper_i, lambda_i, state, step, hess, eye, init_size = carry

                lambda_i = jnp.where(
                    lambda_i > upper_i or lambda_i < lower_i,
                    jnp.max(jnp.array([0.001 * upper_i, jnp.sqrt(upper_i * lower_i)])),
                    lambda_i,
                )

                solver = optx.Cholesky()
                tikhonov = optx.MatrixLinearOperator(
                    hess_mat + lambda_i * eye, optx.positive_semidefinite_tag
                )

                cho_out = optx.linear_solve(tikhonov, state.gradient, solver)
                step, (factor, _) = cho_out.value, cho_out.state

                d_step = optx.linear_solve(factor, step).value

                # Q: should I be passing self as part of the carry to make this
                # a pure function? Same goes for cond fun and atol/rtol.
                phi = ω(step).call(self.norm).ω - delta
                d_phi = -(
                    (ω(d_step).call(lambda x: self.norm(x) ** 2))
                    / ω(step).call(self.norm)
                ).ω

                upper_i = jnp.where(phi < 0, lambda_i, upper_i)
                lower_i = jnp.max(jnp.array([lower_i, lambda_i - phi / d_phi]))

                lambda_i = lambda_i - ((phi - delta) / delta) * (phi / d_phi)

                return (lower_i, upper_i, lambda_i, state, step, hess, eye, init_size)

            _, _, _, _, lm_step, _, _, _ = lax.while_loop(cond_fun, body_fun, init)
            return lm_step

        hess_mat = state.hessian.as_matrix()
        eye = jnp.eye(hess_mat.shape[0])

        newton_step = -optx.linear_solve(state.hessian, state.gradient)

        args = (self, delta, state, newton_step, hess_mat, eye)

        return lax.cond(
            ω(newton_step).call(self.norm).ω < delta, accept_newton, reject_newton, args
        )
