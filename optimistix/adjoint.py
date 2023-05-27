import abc
import functools as ft
from typing import Any, Callable, FrozenSet, Optional

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
        closure: Any,
        tags: FrozenSet[object],
    ):
        ...


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    checkpoints: Optional[int] = None

    def apply(self, primal_fn, rewrite_fn, inputs, closure, tags):
        del rewrite_fn, tags
        while_loop = ft.partial(
            eqxi.while_loop, kind="checkpointed", checkpoints=self.checkpoints
        )
        return primal_fn(inputs, closure, while_loop)


class ImplicitAdjoint(AbstractAdjoint):
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=True)

    def apply(self, primal_fn, rewrite_fn, inputs, closure, tags):
        while_loop = ft.partial(eqxi.while_loop, kind="lax")
        primal_fn = ft.partial(primal_fn, while_loop=while_loop)
        return implicit_jvp(
            primal_fn, rewrite_fn, inputs, closure, tags, self.linear_solver
        )
