from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn
from .._minimise import AbstractMinimiser
from .._misc import filter_cond, tree_where
from .._solution import RESULTS


class _Point(eqx.Module):
    y: Scalar
    f: Scalar


class _GoldenSearchState(eqx.Module):
    lower: _Point
    middle: _Point
    upper: _Point
    first: Bool[Array, ""]


class GoldenSearch(AbstractMinimiser[Scalar, Aux, _GoldenSearchState]):
    """Golden-section search for finding the minimum of a univariate function in a given
    interval.

    This solver requires the following `options`:

    - `lower`: The lower bound on the interval which contains the minimum.
    - `upper`: The upper bound on the interval which contains the minimum.

    This algorithm considers the interval defined by `[lower, upper]` and two additional
    points in between: a `middle` point, and a trial point. If the function value at the
    trial point is lower than at the (previously evaluated) `middle` point, then the
    trial point defines the new `middle` point, and the previous `middle` point defines
    an outer point of the interval containing the minimum.
    In this way, the interval containing the minimum is reduced by the golden ratio at
    each step.

    Note that the initial value `y0` is overwritten in the first step to guarantee that
    the golden ratio is respected throughout. The minimum to be indentified is thus
    defined only by the lower and upper bounds provided.
    """

    rtol: float
    atol: float
    # All norms are the same for scalars.
    norm: ClassVar[Callable[[PyTree], Scalar]] = jnp.abs

    def init(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
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

        # Compute the first mddle point such that the golden ratio is respected.
        # This divides the interval asymmetrically into components A and B, of
        # length a and b, respectively. The ratio of their lengths, b / a, is equal
        # to the golden ratio.
        golden_ratio = (1 + jnp.sqrt(5)) / 2
        middle = (upper - lower) / (golden_ratio + 1)

        f_lower, _ = fn(lower, args)
        f_middle, _ = fn(middle, args)
        f_upper, _ = fn(upper, args)

        return _GoldenSearchState(
            lower=_Point(lower, f_lower),
            middle=_Point(middle, f_middle),
            upper=_Point(upper, f_upper),
            first=jnp.array(True),
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSearchState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _GoldenSearchState, Aux]:
        # Safeguard to ensure that the first step respects the golden ratio rule.
        y_ = jnp.where(state.first, state.lower.y + (state.upper.y - state.middle.y), y)
        f, aux = fn(y_, args)

        # y is either a new candidate minimum point (if the function value at `y_` is
        # lower than elsewhere), or it becomes an outer point, either the lower or the
        # upper bound to the interval in which we keep searching for the minimum. Which
        # bound is replaced depends on whether the current `y_` is higher or lower than
        # the current middle point.
        is_min = f < state.middle.f
        is_lower = y_ < state.middle.y

        def new_middle(point__state):
            point, state = point__state
            new_lower = tree_where(is_lower, state.lower, state.middle)
            new_upper = tree_where(is_lower, state.middle, state.upper)
            return _GoldenSearchState(
                lower=new_lower, middle=point, upper=new_upper, first=jnp.array(False)
            )

        def new_outer(point__state):
            point, state = point__state
            new_lower = tree_where(is_lower, point, state.lower)
            new_upper = tree_where(is_lower, state.upper, point)
            return _GoldenSearchState(
                lower=new_lower,
                middle=state.middle,
                upper=new_upper,
                first=jnp.array(False),
            )

        point = _Point(y_, f)
        new_state = filter_cond(is_min, new_middle, new_outer, (point, state))
        new_y = new_state.lower.y + (new_state.upper.y - new_state.middle.y)

        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSearchState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        y_diff = y - state.middle.y  # This is always the smallest distance
        # Note: currently avoiding computation of f_diff, to avoid calling it here
        # + adding to compilation costs. These could easily be circumvented by writing
        # the value of fn into the solver state. Taking the function values into account
        # when checking convergence might make a difference, hence this is a TODO.

        converged = jnp.abs(y_diff) < self.atol + self.rtol * jnp.abs(y)
        return converged, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSearchState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Scalar, Aux, dict[str, Any]]:
        return y, aux, {}
