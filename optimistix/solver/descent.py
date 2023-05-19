from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractProxyDescent
from ..linear_operator import AbstractLinearOperator
from ..linear_solve import AutoLinearSolver, linear_solve
from ..misc import tree_inner_prod, two_norm
from ..solution import RESULTS


class GradientState(eqx.Module):
    vector: PyTree[Array]


class NewtonState(eqx.Module):
    vector: PyTree[Array]
    operator: AbstractLinearOperator


class UnnormalisedGradient(AbstractProxyDescent[GradientState]):
    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return GradientState(vector)

    def update_state(
        self,
        descent_state: GradientState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        options: Optional[dict[str, Any]] = None,
    ):
        return GradientState(vector)

    def __call__(
        self,
        delta: Scalar,
        descent_state: GradientState,
        args: Any,
        options: dict[str, Any],
    ):
        diff = (-delta * ω(descent_state.vector)).ω
        return diff, jnp.array(RESULTS.successful)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: GradientState,
        args: PyTree,
        options: Optional[dict[str, Any]],
    ):
        return tree_inner_prod(descent_state.vector, diff)


class UnnormalisedNewton(AbstractProxyDescent[NewtonState]):
    gauss_newton: bool = False

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return NewtonState(vector, operator)

    def update_state(
        self,
        descent_state: NewtonState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        options: Optional[dict[str, Any]] = None,
    ):
        return NewtonState(vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: NewtonState,
        args: Any,
        options: dict[str, Any],
    ):
        out = (
            -delta
            * ω(
                linear_solve(
                    descent_state.operator,
                    descent_state.vector,
                    AutoLinearSolver(well_posed=False),
                )
            )
        ).ω
        return out.value, jnp.array(RESULTS.successful)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: NewtonState,
        args: PyTree,
        options: Optional[dict[str, Any]],
    ):
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


class NormalisedGradient(AbstractProxyDescent[GradientState]):
    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return GradientState(vector)

    def update_state(
        self,
        descent_state: GradientState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        options: Optional[dict[str, Any]] = None,
    ):
        return GradientState(vector)

    def __call__(
        self,
        delta: Scalar,
        descent_state: GradientState,
        args: Any,
        options: dict[str, Any],
    ):
        diff = ((-delta * descent_state.vector**ω) / two_norm(descent_state.vector)).ω
        return diff, jnp.array(RESULTS.successful)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: GradientState,
        args: PyTree,
        options: Optional[dict[str, Any]],
    ):
        return tree_inner_prod(descent_state.vector, diff)


class NormalisedNewton(AbstractProxyDescent[NewtonState]):
    gauss_newton: bool = False

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return NewtonState(vector, operator)

    def update_state(
        self,
        descent_state: NewtonState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        options: Optional[dict[str, Any]] = None,
    ):
        return NewtonState(vector, operator)

    def __call__(
        self,
        delta: Scalar,
        descent_state: NewtonState,
        args: Any,
        options: dict[str, Any],
    ):
        out = linear_solve(
            descent_state.operator,
            descent_state.vector,
            solver=AutoLinearSolver(well_posed=False),
        )
        newton = out.value
        diff = ((-delta * newton**ω) / two_norm(newton)).ω
        return diff, jnp.array(RESULTS.successful)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: NewtonState,
        args: PyTree,
        options: Optional[dict[str, Any]],
    ):
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
