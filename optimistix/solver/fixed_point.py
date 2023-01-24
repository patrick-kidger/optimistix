from typing import Callable

import jax.numpy as jnp
from equinox.internal import ω

from ..custom_types import Scalar
from ..fixed_point import AbstractFixedPointSolver
from ..misc import max_norm
from ..solution import RESULTS


class FixedPointIteration(AbstractFixedPointSolver):
    rtol: float
    atol: float
    norm: Callable = max_norm

    def init(self, fixed_point_prob, y, args, options) -> Scalar:
        del fixed_point_prob, y, args, options
        return jnp.array(jnp.inf)

    def step(self, fixed_point_prob, y, args, options, state: Scalar):
        new_y, aux = fixed_point_prob.fn(y, args)
        error = (y**ω - new_y**ω).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        new_state = self.norm((error**ω / scale**ω).ω)
        return new_y, new_state, aux

    def terminate(self, fixed_point_fn, y, args, options, state: Scalar):
        return state < 1, RESULTS.successful
