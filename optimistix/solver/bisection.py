import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Scalar

from ..root_find import AbstractRootFinder
from ..solution import RESULTS


class _BisectionState(eqx.Module):
    lower: Scalar
    upper: Scalar
    error: Scalar
    flip: Bool[Array, ""]


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
        upper_sign = jnp.sign(upper)
        lower_sign = jnp.sign(lower)
        upper_sign = eqxi.error_if(
            upper_sign,
            upper_sign == lower_sign,
            "The root is not contained in [lower, upper]",
        )
        return _BisectionState(
            lower=lower, upper=upper, error=jnp.asarray(jnp.inf), flip=upper_sign < 0
        )

    def step(self, root_prob, y: Scalar, args, options, state):
        del y, options
        new_y = state.lower + 0.5 * (state.upper - state.lower)
        error, aux = root_prob.fn(new_y, args)
        too_large = state.flip ^ (error > 0)
        new_lower = jnp.where(too_large, new_y, state.lower)
        new_upper = jnp.where(too_large, state.upper, new_y)
        new_state = _BisectionState(
            lower=new_lower, upper=new_upper, error=error, flip=state.flip
        )
        return new_y, new_state, aux

    def terminate(self, root_prob, y: Scalar, args, options, state):
        del root_prob, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        return jnp.abs(state.error) < scale, RESULTS.successful

    def buffers(self, state: _BisectionState):
        return ()
