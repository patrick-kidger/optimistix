import abc
from typing import Any, Callable

import equinox as eqx
from jaxtyping import Array, PyTree

from .internal import implicit_jvp
from .linear_operator import Pattern
from .linear_solve import AbstractLinearSolver, AutoLinearSolver


class AbstractAdjoint(eqx.Module):
    @abc.abstractmethod
    def apply(
        self,
        primal_fn: Callable,
        rewrite_fn: Callable,
        inputs: PyTree[Array],
        closure: Any,
        pattern: Pattern,
    ):
        ...


class DirectAdjoint(AbstractAdjoint):
    def apply(self, primal_fn, rewrite_fn, inputs, closure, pattern):
        del rewrite_fn, pattern
        return primal_fn(inputs, closure)


class ImplicitAdjoint(AbstractAdjoint):
    linear_solver: AbstractLinearSolver = AutoLinearSolver()

    def apply(self, primal_fn, rewrite_fn, inputs, closure, pattern):
        return implicit_jvp(
            primal_fn, rewrite_fn, inputs, closure, pattern, self.linear_solver
        )
