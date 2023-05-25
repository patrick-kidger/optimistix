from typing import Any, Tuple

import equinox as eqx
import jax.numpy as jnp
from typing_extensions import TypeAlias

from ..minimise import AbstractMinimiser


OptaxClass: TypeAlias = Any


class OptaxMinimiser(AbstractMinimiser):
    optax_cls: OptaxClass
    args: Tuple[Any, ...]
    kwargs: dict[str, Any]
    max_steps: int

    def __init__(self, optax_cls, *args, max_steps, **kwargs):
        self.optax_cls = optax_cls
        self.args = args
        self.kwargs = kwargs
        self.max_steps = max_steps

    def init(self, problem, y, args, options, aux_struct, f_struct):
        del problem, args, options, aux_struct, f_struct
        step_index = jnp.array(0)
        optim = self.optax_cls(*self.args, **self.kwargs)
        opt_state = optim.init(y)
        return step_index, opt_state

    def step(self, problem, y, args, options, state):
        del options

        @eqx.filter_grad
        def compute_grads(_y):
            value = problem.fn(_y, args)
            if problem.has_aux:
                value, aux = value
            else:
                aux = None
            return value, aux

        grads, aux = compute_grads(y)
        step_index, opt_state = state
        optim = self.optax_cls(*self.args, **self.kwargs)
        updates, new_opt_state = optim.update(grads, opt_state)
        new_y = eqx.apply_updates(y, updates)
        new_state = (step_index + 1, new_opt_state)
        return new_y, new_state, aux

    def terminate(self, problem, y, args, options, state):
        del problem, y, args, options
        step_index, _ = state
        return step_index > self.max_steps
