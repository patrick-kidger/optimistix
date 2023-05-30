import abc
import functools as ft
from typing import Callable, FrozenSet, Optional

import equinox as eqx
import equinox.internal as eqxi
import lineax as lx
from jaxtyping import PyTree

from .ad import implicit_jvp


class AbstractAdjoint(eqx.Module):
    @abc.abstractmethod
    def apply(
        self,
        primal_fn: Callable,
        rewrite_fn: Callable,
        inputs: PyTree,
        tags: FrozenSet[object],
    ):
        ...


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    checkpoints: Optional[int] = None

    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        del rewrite_fn, tags
        while_loop = ft.partial(
            eqxi.while_loop, kind="checkpointed", checkpoints=self.checkpoints
        )
        return primal_fn(inputs, while_loop)


def _primal_fn(inputs):
    primal_fn, inputs, while_loop = inputs
    return primal_fn(inputs, while_loop)


class ImplicitAdjoint(AbstractAdjoint):
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)

    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        _inputs = (primal_fn, inputs, ft.partial(eqxi.while_loop, kind="lax"))
        return implicit_jvp(_primal_fn, rewrite_fn, _inputs, tags, self.linear_solver)
