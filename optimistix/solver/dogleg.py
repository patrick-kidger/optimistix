from typing import Any, Callable, Optional

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractDescent
from ..misc import tree_full_like, tree_inner_prod, tree_where, two_norm


def _quadratic_solve(a, b, c):
    discriminant = jnp.sqrt(b**2 - 4 * a * c)
    # if the output is >2 then we should be accepting the Newton step
    # regardless, so this clip is just a safeguard keeping us in the theoretical
    # bounds.
    return jnp.clip(0.5 * (-b + discriminant) / a, a_min=1, a_max=2)


class DoglegState(eqx.Module):
    vector: PyTree[Array]
    operator: lx.AbstractLinearOperator


class Dogleg(AbstractDescent):
    gauss_newton: bool
    norm: Callable = two_norm
    solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
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
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
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

        numerator = two_norm(grad) ** 2
        denominator = tree_inner_prod(grad, mvp)
        pred = denominator > jnp.finfo(denominator.dtype).eps
        safe_denom = jnp.where(pred, denominator, 1)
        projection_const = jnp.where(pred, numerator / safe_denom, jnp.inf)
        # compute Newton and Cauchy steps. If below Cauchy or above Newton
        # accept (scaled) cauchy or Newton respectively.
        cauchy = (-projection_const * grad**ω).ω
        newton_soln = lx.linear_solve(
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
        b = 2 * tree_inner_prod(cauchy, (newton**ω - cauchy**ω).ω)
        c = cauchy_norm**2 - b - a - delta**2
        linear_interp = _quadratic_solve(a, b, c)
        dogleg = (cauchy**ω + (linear_interp - 1) * (newton**ω - cauchy**ω)).ω
        norm_nonzero = cauchy_norm > jnp.finfo(cauchy_norm.dtype).eps
        safe_norm = jnp.where(norm_nonzero, cauchy_norm, 1)
        normalised_cauchy = tree_where(
            norm_nonzero,
            ((cauchy**ω / safe_norm) * delta).ω,
            tree_full_like(cauchy, jnp.inf),
        )
        diff = tree_where(below_cauchy, normalised_cauchy, newton)
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
