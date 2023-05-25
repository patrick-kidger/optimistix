from typing import Any, Callable, Optional

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractProxyDescent
from ..linear_operator import AbstractLinearOperator
from ..linear_solve import AbstractLinearSolver, AutoLinearSolver, linear_solve
from ..misc import tree_inner_prod, tree_where, two_norm


def _quadratic_solve(a, b, c):
    # oh yeah, we're doing this :)
    discriminant = jnp.sqrt(b**2 - 4 * a * c)
    return 0.5 * (-b + discriminant) / a


class DoglegState(eqx.Module):
    vector: PyTree[Array]
    operator: AbstractLinearOperator


class Dogleg(AbstractProxyDescent):
    gauss_newton: bool
    norm: Callable = two_norm
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=False)

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[AbstractLinearOperator],
        operator_inv: Optional[AbstractLinearOperator],
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        if operator is None:
            assert False
        return DoglegState(vector, operator)

    def update_state(
        self,
        descent_state: DoglegState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[AbstractLinearOperator],
        operator_inv: Optional[AbstractLinearOperator],
        options: Optional[dict[str, Any]] = None,
    ):
        return DoglegState(vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: DoglegState,
        args: Any,
        options: dict[str, Any],
    ):

        if self.gauss_newton:
            # this is just to compute the normalization in the proper direction,
            # which is g^T g (g^T B g)^(-1) where g is J^T r (Jac and residual)
            # and B is J^T J.
            # NOTE: I would not at all be surprised if there was a more efficient
            # way to do this. Look for it!
            grad = descent_state.operator.transpose().mv(descent_state.vector)
            mvp = descent_state.operator.transpose().mv(descent_state.operator.mv(grad))
        else:
            grad = descent_state.vector
            mvp = descent_state.operator.mv(grad)

        # WARNING: CURRENTLY SUFFERING FROM PYTREE ERROR
        numerator = two_norm(grad) ** 2
        denominator = tree_inner_prod(grad, mvp)
        pred = denominator > jnp.finfo(denominator.dtype).eps
        safe_denom = jnp.where(pred, denominator, 1)
        projection_const = jnp.where(pred, numerator / safe_denom, jnp.inf)
        # compute Newton and Cauchy steps. If below Cauchy or above Newton
        # accept (scaled) cauchy or Newton respectively.
        cauchy = (-projection_const * grad**ω).ω
        newton_soln = linear_solve(
            descent_state.operator, descent_state.vector, solver=self.solver
        )
        newton = (-newton_soln.value**ω).ω
        cauchy_norm = self.norm(cauchy)
        newton_norm = self.norm(newton)
        accept_newton = newton_norm <= delta
        below_cauchy = cauchy_norm > delta
        between_choices = jnp.invert(accept_newton) & jnp.invert(below_cauchy)
        # if neither, interpolate between them. We can calculate the exact
        # scalar for this by solving a quadratic equation. See section 4.1 of
        # Nocedal Wright "Numerical Optimization" for details.
        a = two_norm((newton**ω - cauchy**ω).ω) ** 2
        b_ = tree_inner_prod(cauchy, (newton**ω - cauchy**ω).ω)
        b = 2 * b_
        c = cauchy_norm - b_ - a
        linear_interp = _quadratic_solve(a, b, c)
        dogleg = (cauchy**ω + (linear_interp - 1) * (newton**ω - cauchy**ω)).ω
        diff = tree_where(below_cauchy, ((cauchy**ω / cauchy_norm) * delta).ω, newton)
        diff = tree_where(between_choices, dogleg, diff)
        diff = tree_where(accept_newton, newton, diff)
        return diff, newton_soln.result

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: DoglegState,
        args: Any,
        options: dict[str, Any],
    ):
        # same as IndirectIterativeDual
        # The predicted reduction of the iterative dual. This is the model quadratic
        # model function of classical trust region methods localized around f(x).
        # ie. `m(p) = g^t p + 1/2 p^T B p` where `g` is the gradient, `B` the
        # Quasi-Newton approximation to the Hessian, and `p` the
        # descent direction (diff).
        #
        # in the Gauss-Newton setting we compute
        # ```0.5 * [(Jp - r)^T (Jp - r) - r^T r]```
        # which is equivalent when `B = J^T J` and `g = J^T r`.
        if self.gauss_newton:
            rtr = two_norm(descent_state.vector) ** 2
            jacobian_term = (
                two_norm(
                    (ω(descent_state.operator.mv(diff)) - ω(descent_state.vector)).ω
                )
                ** 2
            )
            return 0.5 * (jacobian_term - rtr)
        else:
            operator_quadratic = 0.5 * tree_inner_prod(
                diff, descent_state.operator.mv(diff)
            )
            steepest_descent = tree_inner_prod(descent_state.vector, diff)
            return (operator_quadratic**ω + steepest_descent**ω).ω
