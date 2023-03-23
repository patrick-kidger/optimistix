import abc
import functools as ft
from typing import Any, Callable, FrozenSet, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
from jaxtyping import PyTree

from .ad import implicit_jvp
from .linear_solve import AbstractLinearSolver, AutoLinearSolver


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


def _while_loop(cond_fun, body_fun, init_val, max_steps):
    if max_steps is None:
        return lax.while_loop(cond_fun, body_fun, init_val)
    else:

        def _cond_fun(carry):
            step, val = carry
            return (step < max_steps) & cond_fun(val)

        def _body_fun(carry):
            step, val = carry
            return step + 1, body_fun(val)

        _, final_val = lax.while_loop(_cond_fun, _body_fun, (0, init_val))
        return final_val


class ImplicitAdjoint(AbstractAdjoint):
    linear_solver: AbstractLinearSolver = AutoLinearSolver()

    def apply(self, primal_fn, rewrite_fn, inputs, closure, tags):
        primal_fn = ft.partial(primal_fn, while_loop=_while_loop)
        return implicit_jvp(
            primal_fn, rewrite_fn, inputs, closure, tags, self.linear_solver
        )
