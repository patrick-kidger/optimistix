from collections.abc import Callable
from typing import Generic

import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar, ScalarLike

from .._custom_types import Aux, InequalityOut, Out, Y
from .._misc import max_norm, tree_full_like
from .._quadratic_solve import AbstractQuadraticSolver
from .._search import AbstractDescent, FunctionInfo, Iterate
from .._solution import RESULTS
from .learning_rate import LearningRate


# TODO(jhaffner): The error handling in this solver is hard to read. Clarify!
def _interior_step(
    y: Y, dual: InequalityOut, f_info: FunctionInfo, linear_solver, centrality
) -> tuple[tuple[Y, InequalityOut, InequalityOut], RESULTS]:
    """Constructs and then solves a perturbed KKT system with slack variables system
    from the solver-provided FunctionInfo. Returns an unscaled step, which needs to be
    scaled to its maximum feasible length before being used as a step length.

    Specifically, this solves the linear system

    [ Hess      0    -A^T ] [ y_step ]   [             A^T * l - grad           ]
    [   A      -I      0  ] [ s_step ] = [                   0                  ]
    [   0       L      S  ] [ l_step ]   [ - LSe + centrality * complementarity ]

    in the primal variable `y`, `s` the slack variable, and the dual variable `l`.
    L and S are diagonal matrices of the slack and dual variables, respectively.
    Hess is the Hessian of the target function and A is the Jacobian of the
    constraint function evaluated at y. b is the constraint bound, obtained by
    evaluating the constraint function at y=0.

    The complementarity parameters is defined as the inner product of the slack and dual
    variables, and should be zero at optimality. By adding it to the right-hand side, we
    allow for a relaxation of this condition, which gets tighter as the centrality
    parameter decreases. In this implementation, the centrality parameter is a single
    scalar value that is decreased multiplicatively at each iteration.
    """
    if not isinstance(f_info, FunctionInfo.EvalGradHessian):
        raise ValueError("Interior step requires a gradient and (non-inverse) Hessian.")
    if f_info.constraint_jacobians is None:
        raise ValueError("InteriorPointDescent requires constraints to be defined.")
    else:
        equality_jac, inequality_jac = f_info.constraint_jacobians
        assert equality_jac is None  # Requires small extension to KKT system
    if f_info.constraint_residual is None:
        raise ValueError("Interior step requires a constraint residual.")
    else:
        equality_residual, slack = f_info.constraint_residual
        assert equality_residual is None

        if f_info.bounds is not None:
            raise NotImplementedError(
                "Bounds are not yet supported in InteriorDescent. They may be passed "
                "as inequality constraints instead."
            )

        def make_kkt(hessian, A, dual, slack):  # A: constraint Jacobian
            def kkt(inputs):
                y_step, slack_step, dual_step = inputs

                y_pred = (hessian.mv(y_step) ** ω - A.transpose().mv(dual_step) ** ω).ω
                s_pred = A.mv(y_step) - slack_step
                l_pred = slack * dual_step + dual * slack_step

                return y_pred, s_pred, l_pred

            return kkt

        kkt = make_kkt(f_info.hessian, inequality_jac, dual, slack)
        input_structure = jax.eval_shape(lambda: (y, slack, dual))
        operator = lx.FunctionLinearOperator(kkt, input_structure)

        AT = inequality_jac.transpose()  # pyright: ignore
        complementarity = slack * dual / slack.size
        vector = (
            (AT.mv(dual) ** ω - f_info.grad**ω).ω,
            tree_full_like(slack, 0.0),
            -slack * dual - centrality * complementarity,  # pyright: ignore
        )

        out = lx.linear_solve(operator, vector, linear_solver)
        steps = out.value
        result = RESULTS.promote(out.result)

        return steps, result


# TODO: if this solver survives, change this to use _misc.feasible_step_length
def _truncate_to_feasible(steps, dual, slack, boundary_offset):
    """Truncate the step length to the maximum length that remains strictly feasible:
    both the slack and the dual variables must be greater than zero. Infeasible values
    occur if the magnitude of the computed step exceeds that of the current value, and
    the step is negative.
    """

    def get_feasible_step_size(current, step, boundary_offset):
        step_sizes = jtu.tree_map(
            lambda a, b: jnp.where(b < 0, -a / b, jnp.inf), current, step
        )
        step_sizes, _ = jfu.ravel_pytree(step_sizes)
        return (1 - boundary_offset) * jnp.min(jnp.array([1.0, jnp.min(step_sizes)]))

    y_step, slack_step, dual_step = steps
    primal_step_size = get_feasible_step_size(slack, slack_step, boundary_offset)
    primal_steps = (primal_step_size * (y_step, slack_step) ** ω).ω

    dual_step_size = get_feasible_step_size(dual, dual_step, boundary_offset)
    dual_steps = (dual_step_size * dual_step**ω).ω

    return (*primal_steps, dual_steps)


# TODO: support equality constraints
class _InteriorDescentState(eqx.Module, Generic[Y, InequalityOut], strict=True):
    step: Y
    # Storing dual in descent state: only with LearningRate(1.0) or with eqx.tree_at
    dual: InequalityOut
    centrality: ScalarLike = eqx.field(converter=jnp.asarray)
    result: RESULTS


class InteriorDescent(
    AbstractDescent[
        Y, Iterate.Primal, FunctionInfo.EvalGradHessian, _InteriorDescentState
    ]
):
    """A primal-dual descent through the strict interior of the feasible set.

    Here is what we do: When computing an unconstrained Newton step of a quadratic
    function `x^T Hess x + grad x`, we solve for `Hess x = grad.`
    When solving a constrained quadratic program, we instead consider the Lagrangian
    `x^T Hess x + grad x - lambda * (Ax - b)`, for constraints `Ax = b`, and solve for
    both the primal and dual variables `x` and `lambda`.

    In an interior point descent, we transform inequality constraints of the form
    `Ax >= b` into equality constraints by introducing slack variables `s` such that
    `Ax - s = b`, and use these to constrain the step such that we follow a path that
    takes us through the interior of the feasible set, rather than along the boundary.

    See Nocedal, Wright: "Numerical Optimization", section 16.6.

    !!! warning
        This descent requires and maintains strictly feasible steps in the primal and
        dual variables, and it may fail if this property is not respected. It should not
        be paired with a search that may return a step size greater than one, such as
        [optimistix.ClassicalTrustRegion][], or with a backtracking line search if the
        feasible set is not convex.
    """

    linear_solver: lx.AbstractLinearSolver = lx.SVD()
    boundary_offset: ScalarLike = 1e-3  # TODO: see note on this at the top of the file.

    def init(  # pyright: ignore (for now! TODO)
        self, y: Y, f_info_struct: FunctionInfo.EvalGradHessian
    ) -> _InteriorDescentState:
        assert f_info_struct.constraint_residual is not None
        equality_residual, inequality_residual = f_info_struct.constraint_residual
        assert equality_residual is None
        # TODO: a linear least-squares solve for the initial values of the dual variable
        # would be better, but the required values (f_info.grad, for instance) are not
        # available here.
        dual = tree_full_like(inequality_residual, 1.0)
        # TODO: Make initial value of the centrality parameter a descent attribute
        return _InteriorDescentState(y, dual, 0.9, RESULTS.successful)

    def query(  # pyright: ignore (for now! TODO)
        self, y: Y, f_info: FunctionInfo.EvalGradHessian, state: _InteriorDescentState
    ) -> _InteriorDescentState:
        steps, result = _interior_step(
            y, state.dual, f_info, self.linear_solver, state.centrality
        )
        _, slack = f_info.constraint_residual  # pyright: ignore
        feasible_steps = _truncate_to_feasible(
            steps, state.dual, slack, self.boundary_offset
        )
        y_step, _, dual_step = feasible_steps
        dual = (state.dual**ω + dual_step**ω).ω
        return _InteriorDescentState(y_step, dual, 0.8 * state.centrality, result)

    def step(  # pyright: ignore (for now! TODO)
        self, step_size: Scalar, state: _InteriorDescentState
    ) -> tuple[Y, RESULTS]:
        return (step_size * state.step**ω).ω, state.result


InteriorDescent.__init__.__doc__ = """**Arguments:**

- `linear_solver`: The linear solver to use to compute the interior step.
- `boundary_offset`: The minimum distance to keep to any constraint boundary.
"""


class InteriorPoint(AbstractQuadraticSolver[Y, Out, Aux], strict=True):
    """An interior-point solver for quadratic programs with inequality constraints. Does
    not use a line-search, but will truncate all steps primal and dual steps to the
    maximum length that still respects primal and dual feasibility. Maintains strict
    feasibility throughout the solve.

    Can be used as part of a sequential quadratic program within the descent of a higher
    level solver, such as [`optimistix.BFGS`][].

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: InteriorDescent
    search: LearningRate

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = InteriorDescent(linear_solver=linear_solver)
        self.search = LearningRate(1.0)


InteriorPoint.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm to use for convergence checking.
- `linear_solver`: The linear solver to use when solving for the interior step.
"""
