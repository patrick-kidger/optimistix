import abc
from typing import Any, ClassVar, Dict, Optional, TypeVar

import equinox as eqx
from jaxtyping import ArrayLike, PyTree

from .custom_types import sentinel
from .minimise import AbstractMinimiser, MinimiseProblem


_SearchState = TypeVar("_SearchState")


class AbstractModel(eqx.Module):
    gauss_newton: bool | ClassVar[bool]

    @abc.abstractmethod
    def descent_dir(
        self,
        delta: float,
        problem: MinimiseProblem,
        y: PyTree[ArrayLike],
        state: _SearchState,
        args: PyTree = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> PyTree[ArrayLike]:
        ...


class AbstractTRModel(AbstractModel):
    gauss_newton: bool | ClassVar[bool]

    def __call__(self, x, state):
        ...


class AbstractGLS(AbstractMinimiser):
    needs_gradient: ClassVar[bool] = sentinel
    needs_hessian: ClassVar[bool] = sentinel
    pass
