from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, PyTree, Scalar, ScalarLike

from .._custom_types import Y
from .._misc import filter_cond, two_norm
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


class _FilteredState(eqx.Module, strict=True):
    step_size: Scalar
    filter_by: Float[Array, "objective constraint_violation"]


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

    buffer_size: int = 2**8
    norm: Callable[[PyTree], Scalar] = two_norm
    decrease_factor: ScalarLike = 0.5
    slope: ScalarLike = 0.1
    step_init: ScalarLike = 1.0
    constraint_weight: ScalarLike = 1e-5  # gamma_phi in IPOPT, in (0, 1)
    constraint_decrease: ScalarLike = 1e-5  # gamma_theta in IPOPT, in (0, 1)
    maximum_violation: ScalarLike = 1e-3  # theta_max in IPOPT - what to default to?
    acceptable_violation: ScalarLike = 1e-4  # theta_min in IPOPT
    scale_constraint: ScalarLike = 1.0  # delta in IPOPT
    power_merit: ScalarLike = 2.3  # s_phi in IPOPT
    power_constraint: ScalarLike = 1.1  # s_theta in IPOPT

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
            filter_by=previous_values,
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
            # Subtract bounds term to the merit function.
            # This requires a simple addition of a sum of logarithms to the merit
            # function. However, I do not yet support bounds in the InteriorDescent, so
            # it does not make sense to implement this yet.
            # (I could alternatively just raise a warning.)
            # The one change that would be required is that the log term is scaled by
            # the barrier parameter, which the search does not "know" about. I think it
            # would be best to handle this in the solver, actually - as a function
            # transformation. We also need the gradient of the merit function, and we
            # can only do everything in here because I ignore bounds so far.
            # This would be a good example of a SpaceTransformation, actually.
            raise NotImplementedError("Bounds are not yet supported.")

        if (
            f_info.constraint_residual is None
            or f_eval_info.constraint_residual is None
        ):
            raise ValueError("FilteredLineSearch requires a constraint_residual.")
        else:
            violated_constraints = jnp.where(
                f_info.constraint_residual < 0,
                f_info.constraint_residual,
                0,
            )
            constraint_violation = self.norm(violated_constraints)
            violated_constraints_eval = jnp.where(
                f_eval_info.constraint_residual < 0,
                f_eval_info.constraint_residual,
                0,
            )
            constraint_violation_eval = self.norm(violated_constraints_eval)

            neglect_constraints = constraint_violation < self.acceptable_violation
            scaled_violation = self.scale_constraint * (
                constraint_violation**self.power_constraint
            )
            step = (y_eval**ω - y**ω).ω
            grad_dot = f_info.compute_grad_dot(step)
            # Scaled grad dot is positive if we have a descent direction
            scaled_grad_dot = (-grad_dot) ** self.power_merit
            pred = neglect_constraints & (scaled_grad_dot > scaled_violation)

            # TODO: We might want to make the filter its own small, callable module.
            def filter_(current_values_, augment):
                compare = jnp.all(current_values_[None, :] >= state.filter_by, axis=1)
                filter_out = jnp.any(compare, axis=0)
                update_at = jnp.max(jnp.argmax(state.filter_by, axis=0))
                updated = state.filter_by.at[update_at].set(current_values_)
                filter_by = jnp.where(augment, updated, state.filter_by)
                return jnp.invert(filter_out), filter_by

            def armijo(current_values_):
                _, merit_min_eval_ = current_values_
                predicted_reduction = self.slope * grad_dot
                armijo_ = merit_min_eval_ <= f_info.f + predicted_reduction
                filtered, new_filter = filter_(current_values_, jnp.invert(armijo_))
                return armijo_ & filtered, new_filter

            def improve(current_values_):  # Checks if get a least a little better
                constraint_violation_eval_, merit_min_eval_ = current_values_
                improves_constraints = (
                    constraint_violation_eval_
                    <= (1 - self.constraint_decrease) * constraint_violation
                )
                improves_merit = (
                    merit_min_eval_
                    <= f_info.f - self.constraint_weight * constraint_violation
                )
                improves_either = improves_constraints | improves_merit
                filtered, new_filter = filter_(current_values_, True)  # Always augment
                return improves_either & filtered, new_filter

            current_values = jnp.array([constraint_violation_eval, f_eval_info.f])
            accept, filter_by = filter_cond(pred, armijo, improve, current_values)

            # TODO: special-case the first step: initialise the filter and accept

            step_size = jnp.where(
                accept, self.step_init, self.decrease_factor * state.step_size
            )

            # TODO: check if the step size is accepted and the step size and meets the
            # minimum required step length, otherwise return a rescue request.

            return (
                step_size,
                accept,
                RESULTS.successful,
                _FilteredState(
                    step_size=step_size,
                    filter_by=filter_by,
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
