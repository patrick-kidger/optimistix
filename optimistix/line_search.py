import abc
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar

from .iterate import AbstractIterativeProblem
from .least_squares import LeastSquaresProblem
from .linear_operator import AbstractLinearOperator
from .misc import two_norm


_DescentState = TypeVar("_DescentState")


class AbstractDescent(eqx.Module, Generic[_DescentState]):
    @abc.abstractmethod
    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        ...

    @abc.abstractmethod
    def update_state(
        self,
        descent_state: _DescentState,
        prev_diff: Optional[PyTree[Array]],
        vector: Optional[PyTree[Array]],
        operator: Optional[AbstractLinearOperator],
        options: Optional[dict[str, Any]] = None,
    ):
        ...

    @abc.abstractmethod
    def __call__(
        self,
        delta: Scalar,
        descent_state: _DescentState,
        args: PyTree,
        options: Optional[Dict[str, Any]],
    ):
        ...


class AbstractProxyDescent(AbstractDescent[_DescentState]):
    @abc.abstractmethod
    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _DescentState,
        args: PyTree,
        options: Optional[Dict[str, Any]],
    ) -> PyTree[Array]:
        ...


class _ToMinimiseFn(eqx.Module):
    residual_fn: Callable
    has_aux: bool

    def __call__(self, y, args):
        out = self.residual_fn(y, args)
        if self.has_aux:
            out, aux = out
            return two_norm(out), aux

        else:
            return two_norm(out)


class OneDimensionalFunction(eqx.Module):
    fn: Callable
    descent: Callable
    y: PyTree[Array]

    def __init__(
        self, problem: AbstractIterativeProblem, descent: Callable, y: PyTree[Array]
    ):
        if isinstance(problem, LeastSquaresProblem):
            self.fn = _ToMinimiseFn(problem.fn, True)
        else:
            self.fn = problem.fn
        self.descent = descent
        self.y = y

    def __call__(self, delta: Float[Array, ""], args: PyTree):
        diff, result = self.descent(delta)
        fn, aux = self.fn((ω(self.y) + ω(diff)).ω, args)
        return fn, (fn**2, diff, aux, result, jnp.array(0.0))
