from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..minimise import AbstractMinimiser, MinimiseProblem
from ..misc import tree_where


def _auxmented(fn, has_aux, x, args):
    if has_aux:
        z, original_aux = fn(x, args)
        aux = (z, original_aux)
    else:
        z = fn(x, args)
        aux = z
    return z, aux


class RunningMinMinimiser(AbstractMinimiser):
    minimiser: AbstractMinimiser

    def init(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: PyTree,
        options: dict[str, Any],
    ):
        auxmented = eqx.Partial(_auxmented, problem.fn, problem.has_aux)
        pipethrough_problem = MinimiseProblem(
            auxmented, has_aux=True, tags=problem.tags
        )
        return self.minimiser.init(pipethrough_problem, y, args, options)

    def step(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: PyTree,
        options: dict[str, Any],
        state: PyTree,
    ):
        # Keep track of the running min and output this. keep track of the running
        # iterate through state instead and manage this by augmenting the auxiliary
        # output.
        minimiser_state, state_y, f_min = state
        auxmented = eqx.Partial(_auxmented, problem.fn, problem.has_aux)
        pipethrough_problem = MinimiseProblem(
            auxmented, has_aux=True, tags=problem.tags
        )
        y_new, minimiser_state_new, (f_val, aux) = self.minimiser.step(
            pipethrough_problem, state_y, args, options, minimiser_state
        )
        new_best = f_val < f_min
        y_min = tree_where(new_best, y_new, y)
        f_min_new = jnp.where(new_best, f_val, f_min)
        new_state = (minimiser_state_new, y_new, f_min_new)
        return y_min, new_state, aux

    def terminate(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: PyTree,
        options: dict[str, Any],
        state: PyTree,
    ):
        minimiser_state, state_y, _ = state
        return self.minimiser.terminate(
            problem, state_y, args, options, minimiser_state
        )

    def buffers(self, state: PyTree):
        minimiser_state, *_ = state
        return self.minimiser.buffers(minimiser_state)
