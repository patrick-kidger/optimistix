from typing import cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, Scalar

from ..root_find import AbstractRootFinder
from ..solution import RESULTS


class _BisectionState(eqx.Module):
    lower: Scalar
    upper: Scalar
    error: Scalar
    flip: Bool[Array, ""]
    step: Int[Array, ""]


class Bisection(AbstractRootFinder):
    rtol: float
    atol: float

    def init(self, root_prob, y: Scalar, args, options, f_struct, aux_struct):
        del f_struct, aux_struct
        upper = options["upper"]
        lower = options["lower"]
        if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function taking a "
                "scalar input"
            )
        out_struct, _ = jax.eval_shape(root_prob.fn, y, args)
        if not isinstance(out_struct, jax.ShapeDtypeStruct) or out_struct.shape != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function producing "
                "a scalar output"
            )
        # This computes a range such that `f(0.5 * (a+b))` is
        # the user-passed `lower` on the first step, and the user
        # passed `upper` on the second step. This saves us from
        # compiling `problem.fn` two extra times in the init.
        range = upper - lower
        extended_upper = upper + range
        extended_range = extended_upper - lower
        extended_lower = lower - extended_range
        return _BisectionState(
            lower=extended_lower,
            upper=extended_upper,
            error=jnp.asarray(jnp.inf),
            flip=jnp.array(False),
            step=jnp.array(0),
        )

    def step(self, root_prob, y: Scalar, args, options, state: _BisectionState):
        del y, options
        new_y = state.lower + 0.5 * (state.upper - state.lower)
        error, aux = root_prob.fn(new_y, args)
        too_large = cast(Bool[Array, ""], state.flip ^ (error < 0))
        too_large = jnp.where(state.step == 0, True, too_large)
        too_large = jnp.where(state.step == 1, False, too_large)
        new_lower = jnp.where(too_large, new_y, state.lower)
        new_upper = jnp.where(too_large, state.upper, new_y)
        flip = jnp.where(state.step == 1, error < 0, state.flip)
        message = "The root is not contained in [lower, upper]"
        step = eqxi.error_if(
            state.step, (state.step == 1) & (state.error * error > 0), message
        )
        new_state = _BisectionState(
            lower=new_lower,
            upper=new_upper,
            error=error,
            flip=flip,
            step=step + 1,
        )
        return new_y, new_state, aux

    def terminate(self, root_prob, y: Scalar, args, options, state):
        del root_prob, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        return jnp.abs(state.error) < scale, RESULTS.successful

    def buffers(self, state: _BisectionState):
        return ()
