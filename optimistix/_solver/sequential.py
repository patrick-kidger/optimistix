from collections.abc import Callable
from typing import Generic

import equinox as eqx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from .._custom_types import Y
from .._misc import max_norm, tree_full_like
from .._quadratic_solve import AbstractQuadraticSolver, quadratic_solve
from .._search import AbstractDescent, FunctionInfo
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .bfgs import AbstractOldBFGS
from .interior_point import InteriorPoint


class _QuadraticSubproblemDescentState(eqx.Module, Generic[Y]):
    step: Y
    result: RESULTS


# TODO(jhaffner): This name... is not great.
class QuadraticSubproblemDescent(
    AbstractDescent[Y, FunctionInfo.EvalGradHessian, _QuadraticSubproblemDescentState]
):
    """Iteratively solves a constrained quadratic sub-problem. That is, it uses the
    FunctionInfo created by a higher-order solver (like BFGS) to construct a quadratic
    subproblem from the Hessian and gradient, and linearized constraints from the
    constraint residual and constraint Jacobian.

    The resulting quadratic subproblem is then solved iteratively, and the result is
    used to compute the step.
    """

    quadratic_solver: AbstractQuadraticSolver

    def init(  # pyright: ignore
        self, y: Y, f_info_struct: FunctionInfo
    ) -> _QuadraticSubproblemDescentState:
        del f_info_struct
        # Dummy values of the right shape
        return _QuadraticSubproblemDescentState(y, RESULTS.successful)

    def query(  # pyright: ignore
        self, y: Y, f_info: FunctionInfo, state: _QuadraticSubproblemDescentState
    ) -> _QuadraticSubproblemDescentState:
        if isinstance(f_info, FunctionInfo.EvalGradHessian):
            # Note: currently not checking if the constraint Jacobian exists - the QP
            # could also only treat bounds as constraints (with the appropriate solver.)
            quadratic_approximation = f_info.to_quadratic()
            linear_constraints = f_info.to_linear_constraints()

            sol = quadratic_solve(
                quadratic_approximation,
                self.quadratic_solver,
                tree_full_like(y, 0.0),  # start with zero deviation from y
                bounds=f_info.bounds,
                constraint=linear_constraints,
            )

            step = (sol.value**ω - y**ω).ω
            return _QuadraticSubproblemDescentState(step, sol.result)
        else:
            raise ValueError(
                "Solving a quadratic subproblem requires gradient and (approximate) "
                "Hessian information. The supported FunctionInfo type is "
                "EvalGradHessian."
            )

    def step(
        self, step_size: Scalar, state: _QuadraticSubproblemDescentState
    ) -> tuple[Y, RESULTS]:
        return (step_size * state.step**ω).ω, state.result


QuadraticSubproblemDescent.__init__.__doc__ = """**Arguments**:

`quadratic_solver`: The quadratic solver to use when computing the step.
"""


class SLSQP(AbstractOldBFGS):
    """SLSQP (Sequential Least Squares Quadratic Programming) algorithm for constrained
    optimisation. Makes a BFGS approximation to the target function and linearises the
    constraints at each iterate, then solves the quadratic subproblem as part of its
    descent.

    SLSQP is a bit of a misnomer - SLSQP this is actually a quasi-Newton method, does
    not use residuals, and hence is not an [optimistix.AbstractLeastSquaresSolver][].
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    verbose: frozenset[str]
    search: BacktrackingArmijo
    descent: QuadraticSubproblemDescent

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = False
        self.verbose = verbose
        self.search = BacktrackingArmijo()
        # TODO: hard-coded hotfix while I figure out good ratios for outer/inner tols
        self.descent = QuadraticSubproblemDescent(InteriorPoint(rtol**2, atol**2))


SLSQP.__init__.__doc__ = """**Arguments**:

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
"""
