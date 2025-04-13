from collections.abc import Callable

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, PyTree, Scalar, ScalarLike

from .._custom_types import Y
from .._misc import filter_cond, two_norm
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


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


class _FilteredState(eqx.Module, strict=True):
    step_size: Scalar
    filter: _Filter


# TODO: Perhaps change the name to IPOPTLikeFilteredLineSearch
# TODO: Implement resetting of the filter when the barrier parameter is updated.
# TODO: Buffer size. Check docstring of __init__ below.
# TODO: there are some elements that this search has in common with BacktrackingArmijo.
# A little further down the line, we could consider an AbstractBacktracking class.
# TODO: consider FunctionInfo flavors other than EvalGradHessian, Eval once this works
# well enough. For now I'm restricting myself to this one case to make sure that I won't
# forget to test the other cases properly later.
# TODO: come up with a test case where the filter becomes very important to check that
# we do all the right things there.
# TODO: once this seach can call for a rescue / request a feasibility restoration, note
# that this is a result that the line search may return, and that will only be acted
# upon by specific solvers (IPOPTLike, for now). Alternatively, other solvers could
# raise a warning or an error if they receive this result. (Building custom solvers is
# something our power users do, not something the regular users do.)
class FilteredLineSearch(
    AbstractSearch[Y, FunctionInfo.EvalGradHessian, FunctionInfo.Eval, _FilteredState],
    strict=True,
):
    """A filtered line search. At every step, the search will evaluate the Armijo
    condition on the merit function of the barrier problem if the constraint violation
    is small enough, or it will evaluate whether the step improves either the objective
    function or the constraint violation.
    In either case, the search will then check if the current step is an improvement
    over the previous steps, and reject the step if it is not.

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

    # TODO: if we always augment the filter, then we actually may need a larger buffer?
    # I think the number of steps the solver counts are just the accepted ones.
    buffer_size: int = 2**6  # TODO hotfix while monkey patching
    norm: Callable[[PyTree], Scalar] = two_norm
    decrease_factor: ScalarLike = 0.5
    slope: ScalarLike = 0.1
    step_init: ScalarLike = 1.0
    constraint_weight: ScalarLike = 1e-5  # gamma_phi in IPOPT, in (0, 1)
    constraint_decrease: ScalarLike = 1e-5  # gamma_theta in IPOPT, in (0, 1)
    # TODO: now allowing large(-ish) maximum violation. If this is set to low, then the
    # solver can get stuck - if the first step takes us out of that cushy region and the
    # subsequent ones are substantial improvements, but not quite there, then we will
    # never step to these better places. It is better to not constrain this too much
    # right off the bat, so that we can then let the filter become more stringent
    # throughout the solve. (Plus I think IPOPT has several thingies for this? They also
    # scale by some value?)
    maximum_violation: ScalarLike = 1e3  # theta_max in IPOPT - what to default to?
    acceptable_violation: ScalarLike = 1e-4  # theta_min in IPOPT
    scale_constraint: ScalarLike = 1.0  # delta in IPOPT
    power_merit: ScalarLike = 2.3  # s_phi in IPOPT
    power_constraint: ScalarLike = 1.1  # s_theta in IPOPT

    minimum_step: ScalarLike = 2 ** (-4)  # Backtrack four times at most (1/16)

    # TODO: BacktrackingArmijo uses __post_init__ instead. Would that be better here?
    # https://docs.kidger.site/equinox/api/module/advanced_fields/#checking-invariants
    def __check_init__(self):
        pass

    def init(self, y: Y, f_info_struct: FunctionInfo.EvalGradHessian) -> _FilteredState:
        del f_info_struct

        # TODO: the filter needs to be initialised somewhere.
        # In IPOPT the maximum value for the filter function it is initialised with
        # 1e4 of the initial objective function value, or 1e4, whichever is larger.
        # This could be done when special-casing the first step.
        previous_values = jnp.broadcast_to(
            jnp.array([1000, self.maximum_violation]), (self.buffer_size, 2)
        )
        return _FilteredState(
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
        state: _FilteredState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _FilteredState]:
        if f_info.bounds is not None:
            # TODO hotfix: Only works if we have nonnegativity constraints.
            log_terms = jtu.tree_map(
                lambda x, b: jnp.where(jnp.isfinite(x), jnp.log(x), 0.0),
                y,
                f_info.bounds,  # bounds never get updated
            )
            log_terms, _ = jfu.ravel_pytree(log_terms)
            barrier_term = jnp.sum(log_terms)
            f = f_info.f - barrier_term  # TODO add multiplier here
            # Or else do this in the solver itself, before creating the function info
            log_terms = jtu.tree_map(
                lambda x, b: jnp.where(jnp.isfinite(x), jnp.log(x), 0.0),
                y_eval,
                f_info.bounds,  # bounds never get updated
            )
            log_terms, _ = jfu.ravel_pytree(log_terms)
            barrier_term_eval = jnp.sum(log_terms)
            f_eval = f_eval_info.f - 0.01 * barrier_term_eval
        else:
            f = f_info.f  # no barrier term without bounds
            f_eval = f_eval_info.f

        # Hotfix for debugging (there seems to be an issue with the barrier term)
        f = f_info.f
        f_eval = f_eval_info.f

        if (
            f_info.constraint_residual is None
            or f_eval_info.constraint_residual is None
        ):
            raise ValueError("FilteredLineSearch requires a constraint_residual.")
        else:
            # TODO: re-enable inequality_constraints
            constraint_violation = self.norm(f_info.constraint_residual)
            constraint_violation_eval = self.norm(f_eval_info.constraint_residual)

            # Decide if the constraint violation may be ignored based on gradient values
            # and the constraint violation at the accepted *previous* step y, and the
            # proposed step y_eval - y.
            neglect_constraints = constraint_violation < self.acceptable_violation
            scaled_violation = self.scale_constraint * (
                constraint_violation**self.power_constraint
            )
            step = (y_eval**ω - y**ω).ω
            grad_dot = f_info.compute_grad_dot(step)  # this should be the gradient
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
                improves_constraints = (
                    constraint_violation_eval_
                    <= (1 - self.constraint_decrease) * constraint_violation
                )
                improves_merit = (
                    merit_min_eval_ <= f - self.constraint_weight * constraint_violation
                )
                improves_either = improves_constraints | improves_merit
                filtered, filter = state.filter(current_values_, True)  # Always augment
                return improves_either & filtered, filter

            # TODO: barrier term gets added here - there must be a better way to do this
            current_values = jnp.array([f_eval, constraint_violation_eval])
            accept, filter = filter_cond(pred, armijo, improve, current_values)

            # TODO: special-case the first step: initialise the filter and accept
            accept = first_step | accept

            step_size = jnp.where(
                accept, self.step_init, self.decrease_factor * state.step_size
            )

            # TODO: check if the step size is accepted and the step size and meets the
            # minimum required step length, otherwise return a rescue request.
            result = RESULTS.where(
                step_size >= self.minimum_step,
                RESULTS.successful,
                RESULTS.feasibility_restoration_required,
            )
            # TODO: what should I do with the accept? This determines what branch we
            # end up in. I think I don't need to do anything with it (?), since the
            # step size would fall below the threshold when we have rejected the step
            # and decrease the step size.

            return (
                step_size,
                accept,
                result,
                _FilteredState(
                    step_size=step_size,
                    filter=filter,
                ),
            )


# TODO: are we copying the filter buffer?
# TODO: switch where and update

# TODO: buffer size is currently defaulting to the number of steps in the top-level APIs
# for unconstrained problems. I have so far observed that some constrained searches can
# require more steps - let's see how that develops and update accordingly. We can
# probably get away with storing fewer values, if we assume that the higher ones are
# eventually all superseded anyway.
FilteredLineSearch.__init__.__doc__ = """**Arguments**:

- `buffer_size`: The number of previous values to keep track of in the filter. Default
    value is 256, the same as the default number of steps.
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
"""
