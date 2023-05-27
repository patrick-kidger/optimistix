from typing import Callable

import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from ..fixed_point import AbstractFixedPointSolver
from ..misc import max_norm
from ..solution import RESULTS


class FixedPointIteration(AbstractFixedPointSolver):
    rtol: float
    atol: float
    norm: Callable = max_norm

    def init(self, fixed_point_prob, y, args, options, aux_struct, f_struct) -> Scalar:
        del fixed_point_prob, y, args, options
        return jnp.array(jnp.inf)

    def step(self, fixed_point_prob, y, args, options, state: Scalar):
        new_y, aux = fixed_point_prob.fn(y, args)
        error = (y**ω - new_y**ω).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        new_state = self.norm((error**ω / scale**ω).ω)
        import jax

        jax.debug.print("new_y: {}", new_y)
        return new_y, new_state, aux

    def terminate(self, fixed_point_fn, y, args, options, state: Scalar):
        return state < 1, RESULTS.successful

    def buffers(self, state: PyTree):
        return ()
