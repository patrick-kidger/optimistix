import abc
from typing import ClassVar, TypeVar

import equinox as eqx
import jax

from .custom_types import sentinel
from .minimise import AbstractMinimiser


_SearchState = TypeVar("_SearchState")


class AbstractModel(eqx.Module):
    @abc.abstractmethod
    def descent_dir(self, delta: float, state: _SearchState):
        ...


class AbstractTRModel(AbstractModel):
    def __call__(self, x, state):
        ...


class AbstractQuasiNewtonTR(AbstractTRModel):
    def __call__(self, x, state):
        (grad_flat, _) = jax.flatten_util.tree_ravel(state.grad)
        return state.f_new + state.grad @ x + 0.5 * x.T @ state.hessian.mv(x)


class AbstractGLS(AbstractMinimiser):
    needs_gradient: ClassVar[bool] = sentinel
    needs_hessian: ClassVar[bool] = sentinel
    pass
