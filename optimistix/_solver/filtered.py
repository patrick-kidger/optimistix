from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, PyTree, Scalar, ScalarLike

from .._custom_types import Y
from .._misc import filter_cond, max_norm, tree_dot
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS
from .barrier import LogarithmicBarrier


class _Filter(eqx.Module, strict=True):
    """Implements a filter for the filtered line search. The filter is a buffer that
    contains previous values of the objective function and the constraint violation at
    visited points. By comparing the current values at a given step to the values in the
    filter, the search can reject steps that do not improve the objective function or
    the constraint violation, as compared to previous steps.

    The filter can be augmented. When and why that is done is determined by the search
    that uses it.
    """

    filter_by: Float[Array, "objective constraint_violation"]

    def __call__(self, current_values, augment) -> tuple[Bool[Array, ""], "_Filter"]:
        current_filter = self.filter_by

        compare = jnp.all(current_values[None, :] >= current_filter, axis=1)
        filter_out = jnp.any(compare, axis=0)

        update_at = jnp.max(jnp.argmax(current_filter, axis=0))
        updated = current_filter.at[update_at].set(current_values)
        filter_by = jnp.where(augment, updated, current_filter)

        return jnp.invert(filter_out), _Filter(filter_by)


class _IPOPTLikeFilteredLineSearchState(eqx.Module, strict=True):
    step_size: Scalar
    filter: _Filter


# TODO: consider FunctionInfo flavors other than EvalGradHessian, Eval once this works
# well enough. For now I'm restricting myself to this one case to make sure that I won't
# forget to test the other cases properly later.
class IPOPTLikeFilteredLineSearch(
    AbstractSearch[
        Y,
        FunctionInfo.EvalGradHessian,
        FunctionInfo.Eval,
        _IPOPTLikeFilteredLineSearchState,
    ],
    strict=True,
):
    """A filtered line search. At every step, the search will evaluate the Armijo
    condition on the merit function of the barrier problem if the constraint violation
    is small enough, or it will evaluate whether the step improves either the objective
    function or the constraint violation.
    In either case, the search will then check if the current step is an improvement
    over the previous steps, and reject the step if it is not.

    ??? Differences to IPOPT implementation

    This search is an implementation of the filtered line search available in IPOPT,
    with minimal modifications. All default values for search parameters are set to
    values described in the IPOPT implementation paper, where possible. All core
    features are supported. This includes step acceptance where minimal improvement is
    made in either the merit function or the constraint violation, switching to the
    stricter Armijo condition if the constraint violation is small enough to promote
    faster convergence, and selectively updating the filter, such that it is always
    augmented when the switching condition is not met, and otherwise only augmented if
    the Armijo condition is not met. The search can also request a feasibility
    restoration if the step length becomes too small.

    The modifications are as follows:
    - To respect JAX' static shape requirements, we introduce a buffer size for the
        filter, which defaults to 256 elements, the same as the default number of steps
        in [`optimistix.minimise`][]. If the buffer is full, the largest pairs of (merit
        function, constraint violation) pairs will be booted from the filter, following
        a rule in which the largest value is replaced.
    - The parameters for the maximum and acceptable constraint violation are not set in
        proportion to the initial value of the constraint violation, but instead default
        to their suggested minimum values. Since Optimistix solvers are instantiated
        before any initial values are seen or constraint violations computed, solver
        parameters cannot depend on these.
        If a coupling of solver parameters and initial values of the constraint function
        is desired, the solver can be customised, e.g. with
        ```python
        custom_value = 1e4 * constraint(y0)
        search = IPOPTLikeFilteredLineSearch(maximum_violation=custom_value)
        solver = optx.IPOPTLike(...)
        solver = eqx.tree_at(lambda s: s.search, solver, search)

        solution = optx.minimise(fn, solver, y0, ...)
        ```
        or by defining a custom solver with desired default values.
    - The minimum step size is a constant, defined by the product of `gamma_alpha` and
        `gamma_theta` in IPOPT. The gradients of the merit function and the constraint
        violation are not considered. (In IPOPT these values may be used to accept very
        small steps without resorting to the feasibility restoration, so long as the
        direction is a descent direction.)

    Both of the following two features are rarely invoked in IPOPT according to the
    authors, and are not implemented here:
    - We do not support the "watchdog" feature, which tentantively ignores the filter
        for one iteration. Since previous search states and directions are stored for
        "backup", this would need to be added in the solver that calls this search.
    - We do not support the heuristic filter reset, which resets the filter if a
        proposed step has an acceptable constraint violation, but the previous step
        was rejected by the filter, while tightening filter requirements at each reset.

    ??? cite "References"

        The implementation was developed in:

        ```bibtex
        @article{wachter2006implementation,
        author = {Wächter, Andreas and Biegler, Lorenz T.},
            title = {On the implementation of a primal-dual interior point filter line
                     search algorithm for large-scale nonlinear programming},
            journal = {Mathematical Programming},
            volume = {106},
            number = {1},
            pages = {25-57},
            year = {2006},
            doi = {10.1007/s10107-004-0559-y},
        }
        ```
    """

    buffer_size: int = 2**8
    norm: Callable[[PyTree], Scalar] = max_norm
    decrease_factor: ScalarLike = 0.5  # alpha_red_factor in IPOPT, in (0, 1)
    slope: ScalarLike = 1e-4  # ita_phi in IPOPT, in (0, 0.5)
    step_init: ScalarLike = 1.0  # No specific name in IPOPT, plays role of alpha_0
    constraint_weight: ScalarLike = 1e-5  # gamma_phi in IPOPT, in (0, 1)
    constraint_decrease: ScalarLike = 1e-5  # gamma_theta in IPOPT, in (0, 1)
    maximum_violation: ScalarLike = 1e4  # theta_max in IPOPT
    acceptable_violation: ScalarLike = 1e-4  # theta_min in IPOPT
    scale_constraint: ScalarLike = 1.0  # delta in IPOPT
    power_merit: ScalarLike = 2.3  # s_phi in IPOPT
    power_constraint: ScalarLike = 1.1  # s_theta in IPOPT
    minimum_step_length: ScalarLike = 0.05 * 1e-5  # gamma_alpha * gamma_theta

    def init(
        self, y: Y, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _IPOPTLikeFilteredLineSearchState:
        del f_info_struct

        # TODO: the filter needs to be initialised somewhere.
        # In IPOPT the maximum value for the filter function it is initialised with
        # 1e4 of the initial objective function value, or 1e4, whichever is larger.
        # This could be done when special-casing the first step.
        previous_values = jnp.broadcast_to(
            jnp.array([1000, self.maximum_violation]), (self.buffer_size, 2)
        )
        return _IPOPTLikeFilteredLineSearchState(
            step_size=jnp.array(self.step_init),
            filter=_Filter(previous_values),
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo.EvalGradHessian,
        f_eval_info: FunctionInfo.Eval,
        state: _IPOPTLikeFilteredLineSearchState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _IPOPTLikeFilteredLineSearchState]:
        # TODO: initialise filter if first_step

        # CONSTRUCTION SITE: Barrier transformation ------------------------------------
        f = f_info.f
        f_eval = f_eval_info.f
        if f_info.bounds is not None:
            barrier = LogarithmicBarrier(f_info.bounds)
            f = f_info.f + barrier(y, 1e-2)  # TODO: get barrier parameter from iterate
            f_eval = f_eval_info.f + barrier(y_eval, 1e-2)

            lower_grad, upper_grad = barrier.grads(y, 1e-2)
            f_grad = (f_info.grad**ω + lower_grad**ω + upper_grad**ω).ω
        else:
            f = f_info.f
            f_eval = f_eval_info.f

            f_grad = f_info.grad
        # CONSTRUCTION SITE ENDS -------------------------------------------------------

        if (
            f_info.constraint_residual is None
            or f_eval_info.constraint_residual is None
        ):
            raise ValueError("FilteredLineSearch requires a constraint_residual.")
        else:
            # TODO: this should become an attribute of FunctionInfo
            # It is anyway required for a lot of different solvers, and during
            # termination too.
            equality_residual, inequality_residual = f_info.constraint_residual
            equality_violation = self.norm(equality_residual)
            inequality_violation = self.norm(
                jtu.tree_map(lambda x: jnp.where(x < 0, x, 0.0), inequality_residual)
            )
            constraint_violation = equality_violation + inequality_violation

            (
                equality_residual_eval,
                inequality_residual_eval,
            ) = f_eval_info.constraint_residual
            equality_violation_eval = self.norm(equality_residual_eval)
            inequality_violation_eval = self.norm(
                jtu.tree_map(
                    lambda x: jnp.where(x < 0, x, 0.0), inequality_residual_eval
                )
            )
            constraint_violation_eval = (
                equality_violation_eval + inequality_violation_eval
            )
            ### CONSTRUCTION SITE ENDS -------------------------------------------------

            # Decide if the constraint violation may be ignored based on gradient values
            # and the constraint violation at the accepted *previous* step y, and the
            # proposed step y_eval - y.
            neglect_constraints = constraint_violation < self.acceptable_violation
            scaled_violation = self.scale_constraint * (
                constraint_violation**self.power_constraint
            )
            step = (y_eval**ω - y**ω).ω
            grad_dot = tree_dot(f_grad, step)  # this should be the gradient
            scaled_grad_dot = (-grad_dot) ** self.power_merit
            pred = neglect_constraints & (scaled_grad_dot > scaled_violation)

            def armijo(current_values_):
                merit_min_eval_, _ = current_values_

                predicted_reduction = self.slope * grad_dot
                armijo_ = merit_min_eval_ <= f + predicted_reduction

                filtered, filter = state.filter(current_values_, jnp.invert(armijo_))
                return armijo_ & filtered, filter

            def improve(current_values_):  # Checks if get a least a little better
                merit_min_eval_, constraint_violation_eval_ = current_values_

                constraint_viol_ = (1 - self.constraint_decrease) * constraint_violation
                improves_constraints = constraint_violation_eval_ <= constraint_viol_

                merit_min_ = f - self.constraint_weight * constraint_violation
                improves_merit = merit_min_eval_ <= merit_min_

                improves_either = improves_constraints | improves_merit

                filtered, filter = state.filter(current_values_, True)  # Always augment
                return improves_either & filtered, filter

            current_values = jnp.array([f_eval, constraint_violation_eval])
            accept, filter = filter_cond(pred, armijo, improve, current_values)

            accept = first_step | accept
            step_size = jnp.where(
                accept, self.step_init, self.decrease_factor * state.step_size
            )

            # Invoke feasibility restoration if step length becomes tiny
            result = RESULTS.where(
                step_size >= self.minimum_step_length,
                RESULTS.successful,
                RESULTS.feasibility_restoration_required,
            )

            return (
                step_size,
                accept,
                result,
                _IPOPTLikeFilteredLineSearchState(
                    step_size=step_size,
                    filter=filter,
                ),
            )


IPOPTLikeFilteredLineSearch.__init__.__doc__ = """**Arguments**:

- `buffer_size`: The number of previous values to keep track of in the filter. Default
    value is 256, the same as the default number of steps in [`optimistix.minimise`][].
- `norm`: The norm to use when computing the constraint violation. Should be any 
    function `PyTree -> Scalar`. Optimistix includes three built-in norms: 
    [`optimistix.max_norm`][], [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `decrease_factor`: The rate at which to backtrack, i.e.
    `next_stepsize = decrease_factor * current_stepsize`. Must be between 0 and 1.
- `slope`: The slope of of the linear approximation to `f` that the backtracking 
    algorithm must exceed to terminate. Larger means stricter termination criteria. 
    Must be between 0 and 1.
- `step_init`: The initial step size to use. Default value is 1.0. For constrained 
    solvers, this is reasonable, especially if the descent requires feasible iterates.
- `constraint_weight`: The weight to apply to the constraint violation when checking if
    the step improves the merit function. 
    Default value is 1e-5. Must be between 0 and 1.
- `constraint_decrease`: The factor by which the constraint violation must decrease for
    the step to be considered an improvement. 
    Default value is 1e-5. Must be between 0 and 1.
- `acceptable_violation`: Any constraint violation that is smaller than this value will
    be considered acceptable by the search when determining if the Armijo line 
    search should be used. Default value is 1e-4. Must be non-negative.
- `scale_constraint`: The scaling factor for the constraint violation that must be 
    exceeded by the predicted reduction in the merit function for the step length to be
    computed using the Armijo condition. If this value is smaller, then the solver will 
    prioritize decrease in the merit function, if it is higher then the solver will 
    prioritize minimal constraint violation. Default value is 1.0.
- `power_merit`: The power factor for the merit function that must exceed the constraint
    violation. See `scale_constraint`. 
    Default value is 2.3.
- `power_constraint`: The power factor for the constraint violation that must be 
    exceeded by the predicted reduction in the merit function for the step length to be
    computed using the Armijo condition. See `scale_constraint`. 
    Default value is 1.1.
- `minimum_step_length`: The minimum step length that is considered acceptable. If the
    step length is smaller than this value, the search will request a feasibility 
    restoration. Default value is 0.05 * 1e-5.
"""
