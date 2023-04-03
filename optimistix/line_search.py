import abc
from typing import Any, Callable, Dict, Optional, Tuple

import equinox as eqx
from jaxtyping import Array, ArrayLike, Float, PyTree

from .custom_types import sentinel
from .iterate import AbstractIterativeProblem


class LineSearchState(eqx.Module):
    pass


class AbstractLineSearch(eqx.Module):
    @abc.abstractmethod
    def search(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree,
        search_state: LineSearchState,
        args: PyTree,
        options: Optional[Dict[str, Any]],
        *,
        f_y: Optional[PyTree[Array]] = sentinel,
        gradient: Optional[PyTree[Array]] = sentinel,
        hessian: Optional[PyTree[Array]] = sentinel,
    ) -> Tuple[Float[ArrayLike, " "], LineSearchState]:
        pass


class AbstractTrustRegion(AbstractLineSearch):
    @abc.abstractmethod
    def model_solve(
        self,
        model_fn: Callable,
        y: PyTree,
        search_state: LineSearchState,
        args: PyTree,
        options: Optional[Dict[str, Any]],
        *,
        f_y: Optional[PyTree[Array]] = sentinel,
        gradient: Optional[PyTree[Array]] = sentinel,
        hessian: Optional[PyTree[Array]] = sentinel,
    ) -> Tuple[Float[ArrayLike, " "], LineSearchState]:
        pass


class AbstractModelFunction(eqx.Module):
    @abc.abstractmethod
    def __call__(self, x: PyTree):
        pass
