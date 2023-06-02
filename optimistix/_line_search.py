import abc
from typing import Any, Callable, Generic, Optional, TypeVar

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar

from ._iterate import AbstractIterativeProblem


_DescentState = TypeVar("_DescentState")


class AbstractDescent(eqx.Module, Generic[_DescentState]):
    @abc.abstractmethod
    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: Optional[Any],
        options: Optional[dict[str, Any]],
    ):
        ...

    @abc.abstractmethod
    def update_state(
        self,
        descent_state: _DescentState,
        prev_diff: Optional[PyTree[Array]],
        vector: Optional[PyTree[Array]],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: Optional[dict[str, Any]],
    ):
        ...

    @abc.abstractmethod
    def __call__(
        self,
        delta: Scalar,
        descent_state: _DescentState,
        args: PyTree,
        options: Optional[dict[str, Any]],
    ):
        ...

    @abc.abstractmethod
    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _DescentState,
        args: PyTree,
        options: Optional[dict[str, Any]],
    ) -> PyTree[Array]:
        ...


class OneDimensionalFunction(eqx.Module):
    fn: Callable
    descent: Callable
    y: PyTree[Array]

    def __init__(
        self,
        fn: Callable[[PyTree[Array], PyTree], Scalar],
        descent: Callable,
        y: PyTree[Array],
    ):
        self.fn = fn
        self.descent = descent
        self.y = y

    def __call__(self, delta: Float[Array, ""], args: PyTree):
        diff, result = self.descent(delta)
        new_y = (self.y**ω + diff**ω).ω
        fn, aux = self.fn(new_y, args)
        return fn, (fn, diff, aux, result, jnp.array(0.0))
