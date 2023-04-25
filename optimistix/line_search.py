import abc
from typing import Any, Callable, ClassVar, Dict, Optional, TypeVar

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from .linear_operator import AbstractLinearOperator
from .minimise import AbstractMinimiser, MinimiseProblem


_SearchState = TypeVar("_SearchState")


class AbstractDescent(eqx.Module):
    needs_gradient: ClassVar[bool]
    needs_hessian: ClassVar[bool]

    @abc.abstractmethod
    def __call__(
        self,
        delta: float,
        delta_args: PyTree,
        problem: MinimiseProblem,
        y: PyTree[ArrayLike],
        state: _SearchState,
        args: PyTree,
        options: Optional[Dict[str, Any]],
        vector: PyTree[ArrayLike],
        operator: AbstractLinearOperator,
    ) -> PyTree[ArrayLike]:
        ...


class AbstractProxyDescent(AbstractDescent):
    @abc.abstractmethod
    def predicted_reduction(
        self,
        descent_dir: PyTree[ArrayLike],
        state: _SearchState,
        args: PyTree,
        options: Optional[Dict[str, Any]],
        vector: PyTree[ArrayLike],
        operator: AbstractLinearOperator,
    ) -> PyTree[ArrayLike]:
        ...


class AbstractLineSearch(AbstractMinimiser):
    pass


class OneDimProblem(eqx.Module):
    fn: Callable
    descent_fn: Callable
    y: PyTree[ArrayLike]

    def __call__(self, delta: Float[ArrayLike, " "], args: Any):
        args = (self, delta)
        dynamic_args, static_args = eqx.partition(args, eqx.is_inexact_array)

        def descent_dir(dynamic_args):
            (self, delta) = eqx.combine(dynamic_args, static_args)
            dir, aux = self.descent_fn(delta, None)
            return dir

        def zero_dir(dynamic_args):
            (self, delta) = eqx.combine(dynamic_args, static_args)
            return ω(self.y).call(jnp.zeros_like).ω

        descent_dir = lax.cond(delta != 0.0, descent_dir, zero_dir, dynamic_args)
        fn, aux = self.fn((ω(self.y) + ω(descent_dir)).ω, args)
        return fn, (descent_dir, aux)
