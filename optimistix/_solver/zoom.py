import functools as ft
from typing import ClassVar, Generic, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, Int, Scalar

from .._custom_types import Y
from .._misc import tree_full_like, tree_where, verbose_print
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


IntScalar = Int[Scalar, ""]
FloatScalar = Float[Scalar, ""]
BoolScalar = Bool[Scalar, ""]

# Defining these instead of importing from _search
_FnInfo: TypeAlias = (
    FunctionInfo.EvalGrad
    | FunctionInfo.EvalGradHessian
    | FunctionInfo.EvalGradHessianInv
)


def _cond_print(condition, message, **kwargs):
    """Prints message if condition is true. From Optax."""
    jax.lax.cond(
        condition,
        lambda _: jax.debug.print(message, **kwargs, ordered=True),
        lambda _: None,
        None,
    )


def quadratic_min(
    a: FloatScalar,
    value_a: FloatScalar,
    slope_a: FloatScalar,
    b: FloatScalar,
    value_b: FloatScalar,
) -> FloatScalar:
    """Find the minimum of a quadratic curve fitted to (a, value_a) and (b, value_b)."""
    dist = b - a
    upper = -slope_a * dist**2
    lower = 2 * (value_b - value_a - slope_a * dist)
    return a + upper / lower


def optax_cubicmin(
    a: FloatScalar,
    value_a: FloatScalar,
    slope_a: FloatScalar,
    b: FloatScalar,
    value_b: FloatScalar,
    c: FloatScalar,
    value_c: FloatScalar,
) -> FloatScalar:
    """Cubic interpolation. Adapted from Optax. Optax docs follow:

    Finds a critical point of a cubic polynomial
    p(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D, that goes through the
    points (a,value_a), (b,value_b), and (c,value_c) with derivative at a of slope_a.
    May return NaN (if radical<0), in that case, the point will be ignored.
    Adapted from scipy.optimize._linesearch.py.

    Args:
      a: scalar
      value_a: value of a function f at a
      slope_a: slope of a function f at a
      b: scalar
      value_b: value of a function f at b
      c: scalar
      value_c: value of a function f at c

    Returns:
      xmin: point at which p'(xmin) = 0
    """
    C = slope_a
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    d1 = jnp.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
    A, B = (
        jnp.dot(
            d1,
            jnp.array([value_b - value_a - C * db, value_c - value_a - C * dc]),
            precision=jax.lax.Precision.HIGHEST,
        )
        / denom
    )

    radical = B * B - 3.0 * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (3.0 * A)

    return xmin


def interpolate(
    lo: FloatScalar,
    value_lo: FloatScalar,
    slope_lo: FloatScalar,
    hi: FloatScalar,
    value_hi: FloatScalar,
    cubic_ref: FloatScalar,
    value_cubic_ref: FloatScalar,
) -> FloatScalar:
    """Find a stepsize by minimizing the cubic or quadratic curve.

    Cubic and quadratic curves are fitted to `lo`, `hi`, and `cubic_ref`.
    If the cubic curve's minimum is valid (i.e. sufficiently far from the interval's
    edges), that is used.
    If the cubic minimum is invalid, the quadratic minimum is checked and used if valid.
    If that is also invalid, the middle of the interval is returned.
    """
    # adapted from optax
    delta = jnp.abs(hi - lo)
    left = jnp.minimum(hi, lo)
    right = jnp.maximum(hi, lo)
    cubic_chk = 0.2 * delta  # cubic guess has to be at least this far from the sides
    quad_chk = 0.1 * delta  # quadratic guess has to be at least this far from the sides

    middle_cubic = optax_cubicmin(
        lo, value_lo, slope_lo, hi, value_hi, cubic_ref, value_cubic_ref
    )
    middle_cubic_valid = (middle_cubic > left + cubic_chk) & (
        middle_cubic < right - cubic_chk
    )
    middle_quad = quadratic_min(lo, value_lo, slope_lo, hi, value_hi)
    middle_quad_valid = (middle_quad > left + quad_chk) & (
        middle_quad < right - quad_chk
    )
    middle_bisection = (lo + hi) / 2.0

    a_j = middle_bisection
    a_j = jnp.where(middle_quad_valid, middle_quad, a_j)
    a_j = jnp.where(middle_cubic_valid, middle_cubic, a_j)

    return a_j


class PointEval(eqx.Module, Generic[Y]):
    """Wraps FunctionInfo.Eval including the location."""

    location: Y
    f_info: FunctionInfo.Eval

    @property
    def value(self):
        return self.f_info.f


class PointEvalGrad(eqx.Module, Generic[Y]):
    """Wraps FunctionInfo.EvalGrad including the location"""

    location: Y
    f_info: FunctionInfo.EvalGrad

    def compute_grad_dot(self, y: Y):
        return self.f_info.compute_grad_dot(y)

    def strip_grad(self) -> PointEval[Y]:
        return PointEval(self.location, FunctionInfo.Eval(self.f_info.f))

    @property
    def value(self):
        return self.f_info.f


class ZoomState(eqx.Module, Generic[Y]):
    # number of iterations in the current linesearch
    ls_iter_num: IntScalar
    # point where the linesearch is anchored
    init_point: PointEvalGrad[Y]
    slope_init: FloatScalar
    # last evaluated point
    stepsize: FloatScalar
    current_point: PointEvalGrad[Y]
    current_slope: FloatScalar
    # diagnostics for control flow
    interval_found: BoolScalar
    done: BoolScalar
    failed: BoolScalar
    # interval to zoom into
    stepsize_lo: FloatScalar
    point_lo: PointEvalGrad[Y]
    slope_lo: FloatScalar
    stepsize_hi: FloatScalar
    point_hi: PointEval
    # used for the cubic interpolation
    cubic_ref_stepsize: FloatScalar
    cubic_ref_point: PointEval[Y]
    # fallback stepsize that satisfies at least the decrease condition
    safe_stepsize: FloatScalar
    safe_point: PointEvalGrad[Y]
    safe_slope: FloatScalar
    # used to keep track of the stepsized used for the currently evaluated point
    y_eval_stepsize: FloatScalar
    # descent direction we are taking the steps in from init_point
    descent_direction: Y


class Zoom(AbstractSearch[Y, _FnInfo, FunctionInfo.EvalGrad, ZoomState]):
    """Zoom linesearch.

    Valid entries for `verbose` are:
        - `accepted`: Print when accepting a stepsize.
        - `failed`: Print if the linesearch failed and if the safe stepsize is valid.
        - `linesearch_diagnostics`: Print deeper diagnostics such as which Wolfe
        conditions are satisfied, the interval we're zooming into, if the interval
        is too short, etc.
        - `result_per_ls_iter`: Print linesearch iteration, the checked stepsize,
        and if we're done or failed.
    """

    # TODO decide on defaults
    c1: float = 1e-4
    c2: float = 0.9
    c3: float | None = 1e-6
    max_stepsize: float = 1.0
    increase_factor: float = 1.5
    initial_guess_strategy: str = "keep"
    min_interval_length: float = 1e-6
    min_stepsize: float = 1e-6
    maxls: int = 30
    verbose: frozenset[str] = frozenset()
    _needs_grad_at_y_eval: ClassVar[bool] = True

    def __post_init__(self):
        self.c1 = eqx.error_if(
            self.c1,
            (self.c1 <= 0) | (self.c1 >= 1),
            "`Zoom(c1=...)` must be between 0 and 1.",
        )
        self.c2 = eqx.error_if(
            self.c2,
            (self.c2 <= self.c1) | (self.c2 >= 1),
            "`Zoom(c2=...)` must be between `c1` and 1.",
        )

        self.max_stepsize = eqx.error_if(
            self.max_stepsize,
            self.max_stepsize <= 0,
            "`Zoom(max_stepsize=...)` must be strictly greater than 0.",
        )
        self.increase_factor = eqx.error_if(
            self.increase_factor,
            self.increase_factor <= 0,
            "`Zoom(increase_factor=...)` must be strictly greater than 0.",
        )
        self.min_interval_length = eqx.error_if(
            self.min_interval_length,
            self.min_interval_length <= 0,
            "`Zoom(min_interval_length=...)` must be strictly greater than 0.",
        )
        self.min_stepsize = eqx.error_if(
            self.min_stepsize,
            self.min_stepsize <= 0,
            "`Zoom(min_stepsize=...)` must be strictly greater than 0.",
        )

        if self.c3 is not None:
            self.c3 = eqx.error_if(
                self.c3,
                (self.c3 <= 0) | (self.c3 >= 1),
                "`Zoom(c3=...)` must be between 0 and 1.",
            )

    def init_stepsize_from_previous(self, prev_stepsize: FloatScalar) -> FloatScalar:
        """Initialize the linesearch's stepsize based on the previous steps size.

        Initialization is done aaccording to one of three strategies:
            - "one": initialize to 1.0. Recommended for quasi-Newton methods.
            - "keep": initialize to and start from the previous stepsize.
            - "increase": increase the previous stepsize by `increase_factor`.

        If the initial stepsize would be smaller than the smallest allowed stepsize
        (`min_stepsize`), reset it to `max_stepsize`.

        If the initial stepsize would be larger than the largest allowed stepsize
        (`max_stepsize`), clip it to `max_stepsize`.
        """
        match self.initial_guess_strategy:
            case "one":
                a_i = jnp.array(1.0)
            case "keep":
                a_i = prev_stepsize
            case "increase":
                a_i = prev_stepsize * self.increase_factor
            case _:
                raise ValueError(
                    "initial_guess_strategy has to one of ('one', 'keep', 'increase')"
                )

        # reset if too small
        a_i = jnp.where(a_i <= self.min_stepsize, self.max_stepsize, a_i)
        # guard from above by max_stepsize
        a_i = jnp.minimum(a_i, self.max_stepsize)

        return a_i

    def _actual_init(
        self,
        init_point: PointEvalGrad,
        y_eval_stepsize: FloatScalar,
        y_eval: Y,
    ) -> ZoomState:
        """Init function actually used when starting a new linesearch.

        Called when the number of linesearch steps is reset to 0.

        Instead of initializing the stepsize here, we use the stepsize that was proposed
        at the end of the last linesearch step and was used to create the `y_eval` here.
        """
        # init_point is where stepsize = 0
        descent_direction = (y_eval**ω - init_point.location**ω).ω
        descent_direction = jax.tree.map(
            lambda x: x / y_eval_stepsize,
            descent_direction,
        )
        _slope_init = init_point.compute_grad_dot(descent_direction)

        return ZoomState(
            ls_iter_num=jnp.array(0),
            init_point=init_point,
            slope_init=_slope_init,
            #
            stepsize=jnp.array(0.0),
            current_point=init_point,
            current_slope=_slope_init,
            #
            interval_found=jnp.array(False),
            done=jnp.array(False),
            failed=jnp.array(False),
            #
            stepsize_lo=jnp.array(0.0),
            point_lo=init_point,
            slope_lo=_slope_init,
            stepsize_hi=jnp.array(0.0),
            point_hi=init_point.strip_grad(),
            #
            cubic_ref_stepsize=jnp.array(0.0),
            cubic_ref_point=init_point.strip_grad(),
            #
            safe_stepsize=jnp.array(0.0),
            safe_point=init_point,
            safe_slope=_slope_init,
            #
            y_eval_stepsize=y_eval_stepsize,
            #
            descent_direction=descent_direction,
        )

    def init(self, y, f_info_struct) -> ZoomState:
        """Empty init, called only once when the whole optimization starts.

        Initialize some things to inf to avoid uninitialized values leaking into
        computations without errors.
        """
        del f_info_struct

        _slope_init = jnp.array(-jnp.inf)
        init_point = PointEvalGrad(
            y, FunctionInfo.EvalGrad(jnp.array(jnp.inf), tree_full_like(y, jnp.inf))
        )

        return ZoomState(
            ls_iter_num=jnp.array(0),
            init_point=init_point,
            slope_init=_slope_init,
            #
            stepsize=jnp.array(0.0),
            current_point=init_point,
            current_slope=_slope_init,
            #
            interval_found=jnp.array(False),
            done=jnp.array(False),
            failed=jnp.array(False),
            #
            stepsize_lo=jnp.array(0.0),
            point_lo=init_point,
            slope_lo=_slope_init,
            stepsize_hi=jnp.array(0.0),
            point_hi=init_point.strip_grad(),
            #
            cubic_ref_stepsize=jnp.array(0.0),
            cubic_ref_point=init_point.strip_grad(),
            #
            safe_stepsize=jnp.array(0.0),
            safe_point=init_point,
            safe_slope=_slope_init,
            #
            y_eval_stepsize=jnp.array(-jnp.inf),
            #
            descent_direction=tree_full_like(y, jnp.inf),
        )

    def _propose_by_interpolation(self, state: ZoomState) -> FloatScalar:
        """Propose a stepsize by interpolation, fitting a curve to lo, hi, cubic_ref.

        Function values are matched at all locations and the slope is matched at lo.
        """
        stepsize_middle = interpolate(
            state.stepsize_lo,
            state.point_lo.value,
            state.slope_lo,
            state.stepsize_hi,
            state.point_hi.value,
            state.cubic_ref_stepsize,
            state.cubic_ref_point.value,
        )
        return stepsize_middle

    def _propose_by_increase(self, state: ZoomState) -> FloatScalar:
        """Propose a new stepsize by increasing the current one by `increase_factor`."""
        return state.stepsize * self.increase_factor

    def propose_stepsize(self, state: ZoomState) -> FloatScalar:
        """Propose a stepsize in a way that depends on the zoom linesearch stage.

        If the interval is found and we are zooming into it: interpolate.
        If the interval is not found yet: increase.

        In all cases, limit from above by the maximum allowed stepsize (`max_stepsize`).
        """
        new_stepsize = jax.lax.cond(
            state.interval_found,
            self._propose_by_interpolation,
            self._propose_by_increase,
            state,
        )

        # guard from above by max stepsize
        new_stepsize = jnp.minimum(new_stepsize, self.max_stepsize)

        return new_stepsize

    def decrease_condition_with_approx(
        self,
        stepsize: FloatScalar,
        value_step: FloatScalar,
        slope_step: FloatScalar,
        value_init: FloatScalar,
        slope_init: FloatScalar,
    ) -> BoolScalar:
        """Evaluate the Armijo decrease condition, with an optional approximation.

        If `c3` is not None and the change in function value is sufficiently small
        (with the relative tolerance set by `c3`), that might indicate that the search
        is close to a local minimum. In this case, instead of just checking the decrease
        (Armijo) condition, another condition based on the slope is sufficient for the
        condition to be satisfied.

        Adapted from JAXopt and Optax.
        See [Hager and Zhang, 2006] or the Optax documentation for more information.
        """
        # adopted from jaxopt and optax
        decrease_error = value_step - value_init - self.c1 * stepsize * slope_init
        if self.c3 is not None:
            # if this is >0, the approximate condition is not satisfied
            approx_decrease_error = slope_step - (2 * self.c1 - 1.0) * slope_init
            # if this is >0, the approximate condition will be ignored
            delta_values = value_step - value_init - self.c3 * jnp.abs(value_init)
            # approx_decrease_error <= 0 if both of the above are <= 0
            approx_decrease_error = jnp.maximum(approx_decrease_error, delta_values)
            # near the minimum, the overall condition is satisfied if
            # either the Armijo or the approximate condition are <= 0
            decrease_error = jnp.minimum(approx_decrease_error, decrease_error)

        return decrease_error <= 0.0

    def curvature_condition(
        self, slope_at_new_point: FloatScalar, slope_init: FloatScalar
    ) -> BoolScalar:
        """Evaluate the strong Wolfe curvature condition."""
        curv_error = jnp.abs(slope_at_new_point) - self.c2 * jnp.abs(slope_init)
        return curv_error <= 0.0

    def _zoom_into_interval(self, y, y_eval, f_info, f_eval_info, state) -> ZoomState:
        """Attempt to find an acceptable stepsize in the interval (state.lo, state.lo).

        Shrink the interval if the middle point does not satisfy the conditions.
        """
        del y, f_info

        # y_eval was created by taking state.y_eval_stepsize
        stepsize_middle = state.y_eval_stepsize
        point_middle = PointEvalGrad(y_eval, f_eval_info)
        slope_middle = point_middle.compute_grad_dot(state.descent_direction)

        # check conditions for the middle point
        middle_satisf_decrease = self.decrease_condition_with_approx(
            stepsize_middle,
            point_middle.value,
            slope_middle,
            state.init_point.value,
            state.slope_init,
        )
        middle_satisf_curvature = self.curvature_condition(
            slope_middle, state.slope_init
        )

        if "linesearch_diagnostics" in self.verbose:
            jax.debug.print(
                "Zooming into interval: ({}, {})", state.stepsize_lo, state.stepsize_hi
            )
            jax.debug.print(
                "Middle is: {}\tDecrease: {}\tCurvature: {}",
                stepsize_middle,
                middle_satisf_decrease,
                middle_satisf_curvature,
            )

        middle_lower_than_lo = point_middle.value < state.point_lo.value

        # TODO decide which one to use: largest step or best function value
        # update_safe_stepsize = middle_satisf_decrease & (
        #    point_middle.value < state.safe_point.value
        # )
        update_safe_stepsize = middle_satisf_decrease & (
            stepsize_middle > state.safe_stepsize
        )
        new_safe_stepsize, new_safe_point, new_safe_slope = tree_where(
            update_safe_stepsize,
            [stepsize_middle, point_middle, slope_middle],
            [state.safe_stepsize, state.safe_point, state.safe_slope],
        )

        #
        middle_slope_satisf_third_cond = (
            slope_middle * (state.stepsize_hi - state.stepsize_lo) >= 0
        )

        # new point is not better than lo, so replace the hi side with it, keep lo as lo
        set_hi_to_middle = (~middle_satisf_decrease) | (~middle_lower_than_lo)

        # new point is better than lo, so it will be the new lo
        set_lo_to_middle = middle_satisf_decrease & middle_lower_than_lo
        # same as set_lo_to_middle = not set_hi_to_middle

        # if we overwrite lo with the new point, then we
        # decide which side of the interval to keep based on the third condition
        # if the third condition is satisfied, lo is the new hi
        # otherwise hi stays hi
        set_hi_to_lo = set_lo_to_middle & middle_slope_satisf_third_cond

        # if set_hi_to_lo or set_hi_to_middle, then we overwrite hi
        # and can use it as the reference point
        # otherwise we changed lo, so keep that as reference
        set_cubic_to_hi = set_hi_to_middle | set_hi_to_lo

        # do the updates
        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_middle,
            (stepsize_middle, point_middle.strip_grad()),
            (state.stepsize_hi, state.point_hi),
        )

        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_lo,
            (state.stepsize_lo, state.point_lo.strip_grad()),
            (new_stepsize_hi, new_point_hi),
        )

        new_stepsize_lo, new_point_lo, new_slope_lo = tree_where(
            set_lo_to_middle,
            (stepsize_middle, point_middle, slope_middle),
            (state.stepsize_lo, state.point_lo, state.slope_lo),
        )

        new_cubic_ref, new_cubic_ref_point = tree_where(
            set_cubic_to_hi,
            (state.stepsize_hi, state.point_hi),
            (state.stepsize_lo, state.point_lo.strip_grad()),
        )

        # if middle satisfies both conditions, then we accept it as the final stepsize
        done = middle_satisf_decrease & middle_satisf_curvature

        interval_too_short = (
            jnp.abs(new_stepsize_hi - new_stepsize_lo) <= self.min_interval_length
        )

        # diagnose failure the same way optax does
        max_iter_reached = (state.ls_iter_num + 1) >= self.maxls
        presumably_failed = max_iter_reached | (
            interval_too_short & (new_safe_stepsize > 0.0)
        )
        failed = presumably_failed & (~done)

        if "linesearch_diagnostics" in self.verbose:
            _cond_print(
                interval_too_short,
                "Interval too short: ({ss_lo}, {ss_hi})",
                ss_lo=new_stepsize_lo,
                ss_hi=new_stepsize_hi,
            )

        return ZoomState(
            ls_iter_num=state.ls_iter_num + 1,
            #
            init_point=state.init_point,
            slope_init=state.slope_init,
            #
            stepsize=stepsize_middle,
            current_point=point_middle,
            current_slope=slope_middle,
            #
            stepsize_lo=new_stepsize_lo,
            point_lo=new_point_lo,
            slope_lo=new_slope_lo,
            stepsize_hi=new_stepsize_hi,
            point_hi=new_point_hi,
            #
            interval_found=state.interval_found,
            done=done,
            failed=failed,
            #
            cubic_ref_stepsize=new_cubic_ref,
            cubic_ref_point=new_cubic_ref_point,
            #
            safe_stepsize=new_safe_stepsize,
            safe_point=new_safe_point,
            safe_slope=new_safe_slope,
            #
            y_eval_stepsize=state.y_eval_stepsize,
            #
            descent_direction=state.descent_direction,
        )

    def _search_interval(
        self,
        y: Y,
        y_eval: Y,
        f_info,
        f_eval_info,
        state: ZoomState,
    ):
        """Look for interval to zoom into."""
        del y, f_info

        # evaluate the slope along the descent direction for the new stepsize
        new_stepsize = state.y_eval_stepsize
        new_point = PointEvalGrad(y_eval, f_eval_info)
        slope_at_new_point = new_point.compute_grad_dot(state.descent_direction)

        reached_max_stepsize = new_stepsize >= self.max_stepsize

        # Check the conditions for the new point
        decrease_satisfied = self.decrease_condition_with_approx(
            new_stepsize,
            new_point.value,
            slope_at_new_point,
            state.init_point.value,
            state.slope_init,
        )
        value_increased = new_point.value >= state.current_point.value
        curvature_satisfied = self.curvature_condition(
            slope_at_new_point, state.slope_init
        )

        if "linesearch_diagnostics" in self.verbose:
            jax.debug.print("Searching for interval.")
            jax.debug.print(
                "Checked stepsize: {}, Decrease: {}, Curvature: {}",
                new_stepsize,
                decrease_satisfied,
                curvature_satisfied,
            )

        # save this as the largest stepsize that satisfies the decrease condition
        new_safe_stepsize, new_safe_point, new_safe_slope = tree_where(
            decrease_satisfied,
            (new_stepsize, new_point, slope_at_new_point),
            (state.safe_stepsize, state.safe_point, state.safe_slope),
        )

        # There are two conditions when we say we found an interval
        found_a = (~decrease_satisfied) | (value_increased & (state.ls_iter_num > 0))
        found_b = (~found_a) & (~curvature_satisfied) & (slope_at_new_point >= 0)
        interval_found = found_a | found_b

        # If the interval is found, from the next iteration on we do _zoom_into_interval
        # if found_a: we call zoom(alpha_lo = alpha_{i-1}, alpha_hi = alpha_{i})
        # if found_b: we call zoom(alpha_lo = alpha_{i}, alpha_hi = alpha_{i-1})
        # where state.stepsize is alpha_{i-1} and new_stepsize is alpha_{i}

        # If the interval is found, this will zoom into the correct interval.
        # If not, it still sets lo and hi, but that's okay because it will not be used
        # by _zoom_into_interval, and we will just return here in the next iteration.
        new_stepsize_lo, new_point_lo, new_slope_lo = tree_where(
            found_a,
            (state.stepsize, state.current_point, state.current_slope),
            (new_stepsize, new_point, slope_at_new_point),
        )
        new_stepsize_hi, new_point_hi = tree_where(
            found_a,
            (new_stepsize, new_point.strip_grad()),
            (state.stepsize, state.current_point.strip_grad()),
        )

        # TODO shouldn't we update the reference point?

        # from optax
        done = (decrease_satisfied & curvature_satisfied) | (
            reached_max_stepsize & ~interval_found
        )
        failed = (state.ls_iter_num + 1 >= self.maxls) & (~done)

        return ZoomState(
            ls_iter_num=state.ls_iter_num + 1,
            #
            init_point=state.init_point,
            slope_init=state.slope_init,
            #
            stepsize=new_stepsize,
            current_point=new_point,
            current_slope=slope_at_new_point,
            #
            stepsize_lo=new_stepsize_lo,
            point_lo=new_point_lo,
            slope_lo=new_slope_lo,
            stepsize_hi=new_stepsize_hi,
            point_hi=new_point_hi,
            #
            cubic_ref_stepsize=new_stepsize_lo,
            cubic_ref_point=new_point_lo.strip_grad(),
            #
            interval_found=interval_found,
            done=done,
            failed=failed,
            #
            safe_stepsize=new_safe_stepsize,
            safe_point=new_safe_point,
            safe_slope=new_safe_slope,
            #
            y_eval_stepsize=state.y_eval_stepsize,
            #
            descent_direction=state.descent_direction,
        )

    def fake_first_step(
        self,
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: FunctionInfo.EvalGrad,
        state: ZoomState,
    ):
        """Just accepts the proposed random point.

        Only called once in the very beginning of the optimization.
        """
        del y, y_eval, f_info, f_eval_info
        accept = jnp.array(True)
        return accept, state

    def _safe_step(
        self,
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: FunctionInfo.EvalGrad,
        state: ZoomState,
    ):
        """Called after the search fails and the safe stepsize is proposed,
        `y_eval` was created with that safe stepsize, so just accept it.
        """
        del y, y_eval, f_info, f_eval_info
        accept = jnp.array(True)
        return accept, state

    def _regular_step(
        self,
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: FunctionInfo.EvalGrad,
        state: ZoomState,
    ):
        """This is a proper step of the zoom linesearch.

        Dispatches to either _search_interval or _zoom_into_interval.
        Accepts `y_eval` if those set `state.done` to True.
        """
        if "result_per_ls_iter" in self.verbose:
            jax.debug.print("Zoom iter: {}", state.ls_iter_num)

        _zoom_fn = ft.partial(
            self._zoom_into_interval,
            y,
            y_eval,
            f_info,
            f_eval_info,
        )
        _search_fn = ft.partial(
            self._search_interval,
            y,
            y_eval,
            f_info,
            f_eval_info,
        )

        state = jax.lax.cond(
            state.interval_found,
            _zoom_fn,
            _search_fn,
            state,
        )

        if "result_per_ls_iter" in self.verbose:
            verbose_print(
                (True, "Checked", state.stepsize),
                (True, "Done", state.done),
                (True, "Failed", state.failed),
            )

        # only accept the stepsize we checked here if it's good
        accept = state.done

        return accept, state

    def _step(
        self,
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: FunctionInfo.EvalGrad,
        state: ZoomState,
    ):
        """This is repeatedly called after the first fake step is out of the way.

        Potentially reinitialize the state, then perform a regular step or accept the
        safe stepsize.
        """
        # on the first real iteration of the linesearch, reinitialize the state
        init_point = PointEvalGrad(y, FunctionInfo.EvalGrad(f_info.f, f_info.grad))
        _reinit_state_fn = ft.partial(
            self._actual_init,
            init_point,
            state.y_eval_stepsize,  # proposed at the end of the previous linesearch
            y_eval,  # created with state.y_eval_stepsize
        )
        state = jax.lax.cond(
            state.ls_iter_num == 0,
            _reinit_state_fn,
            lambda: state,
        )

        # if we failed on the previous iteration, y_eval was made with the safe stepsize
        # so just accept it with _safe_step
        # otherwise take a regular step
        _safe_step_fn = ft.partial(self._safe_step, y, y_eval, f_info, f_eval_info)
        _regular_step_fn = ft.partial(
            self._regular_step, y, y_eval, f_info, f_eval_info
        )
        accept, state = jax.lax.cond(
            state.failed,
            _safe_step_fn,
            _regular_step_fn,
            state,
        )

        return accept, state

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: FunctionInfo.EvalGrad,
        state: ZoomState,
    ):
        """Dispatches to fake_first_step if `first_step` is true, to _step otherwise.

        `y_eval` is accepted in 3 conditions:
            - on the fake first step
            - it satisfies both the decrease and the curvature conditions
            - we failed on the previous iteration, `y_eval` was created with the safe
            stepsize, and are accepting that

        If `y_eval` is accepted:
            - reset the linesearch iteration to zero,
            allowing the initialization to be triggered the next time it is called.
            - propose a new stepsize for the next linesearch based on the currently
            accepted final stepsize
        If we failed on the current iteration:
            - if the safe stepsize is useful, replace the stepsize with that
            - propose the (hopefully safe) stepsize to be accepted in the next
            iteration by _safe_step
        """
        if not isinstance(f_info, _FnInfo):
            raise ValueError(
                "Cannnot use `Zoom` with this solver."
                "This is because `Zoom` requires the objective function's gradient"
                "at the last accepted point (`y`)."
                "In other words, the type of variable `f_info` needs to be"
                f"one of {_FnInfo}.".replace(" | ", ", ")
            )
        if not isinstance(f_eval_info, FunctionInfo.EvalGrad):
            raise ValueError(
                "Cannnot use `Zoom` with this solver."
                "This is because `Zoom` requires the objective function's gradient"
                "at the currently evaluated point (`y_eval`)."
                "In other words, variable `f_eval_info` needs to be"
                "of type `FunctionInfo.EvalGrad`."
            )

        _fake_first_step_fn = ft.partial(
            self.fake_first_step, y, y_eval, f_info, f_eval_info
        )
        _step_fn = ft.partial(self._step, y, y_eval, f_info, f_eval_info)

        accept, state = jax.lax.cond(
            first_step,
            _fake_first_step_fn,
            _step_fn,
            state,
        )

        if "accepted" in self.verbose:
            _cond_print(accept & (~first_step), "Accepting {ss}", ss=state.stepsize)

        # if accepted, reset the linesearch iteration counter
        new_ls_iter_num = jnp.where(
            accept,
            jnp.array(0),
            state.ls_iter_num,
        )
        # and propose an initial stepsize for the next linesearch
        proposed_stepsize = jnp.where(
            accept,
            self.init_stepsize_from_previous(state.stepsize),
            self.propose_stepsize(state),
        )

        # if we failed on the current iteration (so failed but didn't accept yet)
        # and the safe stepsize is valid
        # then try backpedaling to the safe step
        # and propose that instead so that we take one more step with it
        # which will be accepted by _safe_step
        # on the next iteration
        try_safe_stepsize = ~accept & state.failed & (state.safe_stepsize > 0.0)
        new_stepsize, new_point, new_slope = tree_where(
            try_safe_stepsize,
            (state.safe_stepsize, state.safe_point, state.safe_slope),
            (state.stepsize, state.current_point, state.current_slope),
        )
        proposed_stepsize = jnp.where(
            try_safe_stepsize,
            state.safe_stepsize,
            proposed_stepsize,
        )

        # TODO what if we fail and the safe stepsize is not valid?
        # Maybe don't return RESULTS.successful?
        if "failed" in self.verbose:
            _cond_print(
                ~accept & state.failed & (state.safe_stepsize <= 0.0),
                "Failed and safe stepsize is invalid.",
            )
            _cond_print(
                ~accept & state.failed & (state.safe_stepsize > 0.0),
                "Failed but proposing safe stepsize.",
            )

        # write these into the state
        state = eqx.tree_at(
            lambda s: (
                s.stepsize,
                s.current_point,
                s.current_slope,
                s.ls_iter_num,
                s.y_eval_stepsize,
            ),
            state,
            (
                new_stepsize,
                new_point,
                new_slope,
                new_ls_iter_num,
                proposed_stepsize,
            ),
        )

        return proposed_stepsize, accept, RESULTS.successful, state


Zoom.__init__.__doc__ = """**Arguments:**
    - `c1`: Relative tolerance in the decrease (Armijo) condition.
    - `c2`: Relative tolerance in the curvature (Wolfe) condition.
    - `c3`: Relative tolerance in function value to trigger the approximate decrease
        condition (from Hager and Zhang, 2006).
        If `None`, the approximate decrease condition is never checked.
    - `max_stepsize`: Largest allowed stepsize.
    - `increase_factor`: Multiplication factor for increasing the stepsize when looking
        for an interval or proposing a new starting stepsize for the next linesearch
        after accepting the current one.
    - `initial_guess_strategy`: Method for initializing the starting stepsize for the
        next linesearch based on the current one's final accepted stepsize.
        Possible values are:
          - "one": initialize to 1.0
          - "keep": use the final stepsize.
          - "increase": increase by `increase_factor`.
    - `min_interval_length`: Stop zooming the interval's length in falls below this.
    - `min_stepsize`: If the initial stepsize would be smaller than this smallest
        allowed stepsize, reset it to `max_stepsize`.
    - `maxls`: Maximum number of linesearch iterations.
"""
