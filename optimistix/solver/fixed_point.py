from typing import Callable

import jax.numpy as jnp
from equinox.internal import ω

from ..custom_types import Scalar
from ..fixed_point import AbstractFixedPointSolver
from ..misc import rms_norm
from ..results import RESULTS


class FixedPointIteration(AbstractFixedPointSolver):
    rtol: float
    atol: float
    norm: Callable = rms_norm

    def init(self, fixed_point_fn, y, args, options) -> Scalar:
        del fixed_point_fn, y, args, options
        return jnp.array(jnp.inf)

    def step(self, fixed_point_fn, y, args, options, state: Scalar):
        y_next = fixed_point_fn(y, args)
        error = (y**ω - y_next**ω).ω
        scale = (self.atol + self.rtol * ω(y_next).call(jnp.abs)).ω
        return y_next, self.norm((error**ω / scale**ω).ω)

    def terminate(self, fixed_point_fn, y, args, options, state: Scalar):
        return state < 1, RESULTS.successful
