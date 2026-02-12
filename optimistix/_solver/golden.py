import math
from typing import Any

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree

from .._custom_types import Aux, Fn
from .._minimise import AbstractMinimiser
from .._misc import tree_where
from .._solution import RESULTS
from .._termination import CauchyTermination


class _GoldenSearchState(eqx.Module):
    lower: Float[Array, ""]
    middle: Float[Array, ""]
    upper: Float[Array, ""]
    f_middle: Float[Array, ""]
    first: Bool[Array, ""]
    terminate: Bool[Array, ""]


class GoldenSearch(AbstractMinimiser[Float[Array, ""], Aux, _GoldenSearchState]):
    """Golden-section search for finding the minimum of a univariate function in a given
    interval. It does not use gradients and is not particularly fast, but it is very
    robust.

    This solver maintains a set of three reference points, defining the lower and upper
    boundaries of the interval, as well as a midpoint chosen to divide the interval into
    two sections by the golden ratio. At each step, the reference points are updated by
    dropping one of the outer points and updating the midpoint, such that the interval
    shrinks monotonously until the solver has converged.

    If the function is unimodal (has just one minimum inside the interval), then this
    minimum is always found. If the function has several minima, then a local minimum is
    identified, depending on the initial choice of interval bounds.

    This solver requires the following `options`:

    - `lower`: The lower bound on the interval which contains the minimum.
    - `upper`: The upper bound on the interval which contains the minimum.

    Note that the initial value `y0` is ignored to guarantee that the golden ratio
    between interval segments is always maintained.
    """

    termination: CauchyTermination

    def __init__(self, rtol: float, atol: float):
        # All norms are the same for scalars.
        self.termination = CauchyTermination(rtol, atol, norm=jnp.abs)

    def init(
        self,
        fn: Fn[Float[Array, ""], Float[Array, ""], Aux],
        y: Float[Array, ""],
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GoldenSearchState:
        del aux_struct
        lower = jnp.asarray(options["lower"], f_struct.dtype)
        upper = jnp.asarray(options["upper"], f_struct.dtype)
        if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
            raise ValueError(
                "GoldenSearch can only be used to find the minimum of a function "
                "taking a scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "GoldenSearch can only be used to find the minimum of a function "
                "producing a scalar output."
            )

        # Compute the first middle point such that the golden ratio is respected.
        # This divides the interval asymmetrically into components A and B, of
        # length a and b, respectively. The ratio of their lengths, b / a, is equal
        # to the golden ratio.
        golden_ratio = (1 + math.sqrt(5)) / 2
        middle = (upper - lower) / (golden_ratio + 1)

        f_middle, _ = fn(middle, args)

        return _GoldenSearchState(
            lower=lower,
            middle=middle,
            upper=upper,
            f_middle=f_middle,
            first=jnp.array(True),
            terminate=jnp.array(False),
        )

    def step(
        self,
        fn: Fn[Float[Array, ""], Float[Array, ""], Aux],
        y: Float[Array, ""],
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSearchState,
        tags: frozenset[object],
    ) -> tuple[Float[Array, ""], _GoldenSearchState, Aux]:
        # Safeguard to ensure that the first step respects the golden ratio rule.
        y_ = jnp.where(state.first, state.lower + (state.upper - state.middle), y)
        del y

        f, aux = fn(y_, args)

        # Decide whether to terminate the solve after this step. We'll still compute a
        # `new_y` below, but if we have converged already then its value is only going
        # to be negligibly different from `y_`. We're comparing against the middle point
        # since that is always the point closest to the current `y_`.
        y_diff = state.middle - y_
        f_diff = state.f_middle - f
        terminate = self.termination(
            state.middle,
            y_diff,
            state.f_middle,
            f_diff,
        )

        # y is either a new candidate minimum point (if the function value at `y_` is
        # lower than elsewhere), or it becomes an outer point, either the lower or the
        # upper bound to the interval in which we keep searching for the minimum. Which
        # bound is replaced depends on whether the current `y_` is higher or lower than
        # the current middle point.
        is_min = f < state.f_middle
        is_lower = y_ < state.middle

        def new_middle(state):
            new_lower = tree_where(is_lower, state.lower, state.middle)
            new_upper = tree_where(is_lower, state.middle, state.upper)
            return _GoldenSearchState(
                lower=new_lower,
                middle=y_,
                upper=new_upper,
                f_middle=f,
                first=jnp.array(False),
                terminate=terminate,
            )

        def new_outer(state):
            new_lower = tree_where(is_lower, y_, state.lower)
            new_upper = tree_where(is_lower, state.upper, y_)
            return _GoldenSearchState(
                lower=new_lower,
                middle=state.middle,
                upper=new_upper,
                f_middle=state.f_middle,
                first=jnp.array(False),
                terminate=terminate,
            )

        new_state = lax.cond(is_min, new_middle, new_outer, state)
        new_y = new_state.lower + (new_state.upper - new_state.middle)

        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Float[Array, ""], Float[Array, ""], Aux],
        y: Float[Array, ""],
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSearchState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Float[Array, ""], Float[Array, ""], Aux],
        y: Float[Array, ""],
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSearchState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Float[Array, ""], Aux, dict[str, Any]]:
        return y, aux, {}


GoldenSearch.__init__.__doc__ = """**Arguments:**

- `rtol`: The relative tolerance for terminating the solve.
- `atol`: The absolute tolerance for terminating the solve.
"""
