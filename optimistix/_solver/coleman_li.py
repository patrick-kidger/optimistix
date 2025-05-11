from collections.abc import Callable
from typing import Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from .._custom_types import Aux, Y
from .._misc import max_norm
from .._search import AbstractDescent, FunctionInfo, Iterate
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .bfgs import AbstractOldBFGS


# TODO: I think the Coleman-Li approach can be slightly simplified (Notes below).
@jax.custom_jvp
def _compute_scaling_pytree(y: Y, f_info: FunctionInfo.EvalGradHessian) -> Y:
    # As described in Coleman & Li.
    def scaling(x, grad, lower, upper):
        has_upper = jnp.isfinite(upper)
        has_lower = jnp.isfinite(lower)
        scaling = jnp.zeros_like(x)
        scaling = jnp.where((grad < 0) & has_upper, (upper - x), scaling)
        scaling = jnp.where((grad >= 0) & has_lower, (x - lower), jnp.asarray(scaling))
        scaling = jnp.where(
            (grad < 0) & ~has_upper, -1, jnp.asarray(scaling)
        )  # Why -1?
        scaling = jnp.where((grad >= 0) & ~has_lower, 1, scaling)
        # This seems like a footgun. If all iterates of y are strictly feasible, then
        # the scaling values will always be positive (if we change the weird -1 two rows
        # above to 1, which is never used anywhere, since D and J(v(x)) both use the
        # absolute value). Might be better to return scaling as is and then assert that
        # it be positive? Otherwise we could be masking problems in the solve, I think.
        return jnp.abs(scaling)

    return jtu.tree_map(scaling, y, f_info.grad, *f_info.bounds)  # pyright: ignore


@_compute_scaling_pytree.defjvp
def _compute_scaling_pytree_jvp(primals, tangents):
    y, f_info = primals
    y_dot, _ = tangents

    def derivative(x, grad, lower, upper):
        has_upper = jnp.isfinite(upper)
        has_lower = jnp.isfinite(lower)
        derivative = jnp.zeros_like(x)
        derivative = jnp.where((grad < 0) & has_upper, -1, derivative)
        derivative = jnp.where((grad >= 0) & has_lower, 1, derivative)
        return derivative

    primal_out = _compute_scaling_pytree(y, f_info)
    tangent_out = jtu.tree_map(
        lambda a, b: a * b,
        jtu.tree_map(derivative, y, f_info.grad, *f_info.bounds),
        y_dot,
    )
    return primal_out, tangent_out


def _interior_reflected_newton_step(
    y: Y, f_info: FunctionInfo, linear_solver: lx.AbstractLinearSolver
) -> tuple[Y, RESULTS]:
    """Compute an interior-reflected Newton step."""
    if not isinstance(f_info, FunctionInfo.EvalGradHessian):
        raise NotImplementedError

    scaling = _compute_scaling_pytree(y, f_info)
    diagonal = lx.DiagonalLinearOperator(scaling)  # pyright: ignore
    gradient = lx.DiagonalLinearOperator(f_info.grad)

    # # Compute the Jacobian: Jac(|v|)
    scaling_jac = jax.jacfwd(_compute_scaling_pytree)(y, f_info)
    out_structure = jax.eval_shape(lambda: y)
    # When the gradient is zero, the row in the Jacobian should be set to zero too -
    # according to Coleman & Li. But since we're taking the product, this is redundant,
    # at least as long as strict feasibility is maintained. (I think?)
    jacobian = lx.PyTreeLinearOperator(scaling_jac, out_structure)

    # TODO: is this operator going to be positive definite? Tag it if so.
    operator = diagonal @ f_info.hessian + gradient @ jacobian
    vector = diagonal.mv(f_info.grad)

    out = lx.linear_solve(operator, vector, linear_solver)

    # Pull back to the original space for compatibility with modular searches.
    pullback = lx.DiagonalLinearOperator(
        jtu.tree_map(lambda x: 1 / jnp.sqrt(x), scaling)
    )
    interior_reflected_newton = pullback.mv(out.value)
    # TODO: truncate to feasible step

    result = RESULTS.promote(out.result)
    return interior_reflected_newton, result


class _ColemanLiDescentState(eqx.Module, Generic[Y], strict=True):
    newton: Y
    result: RESULTS


# Reference: https://link.springer.com/article/10.1007/BF01582221
# TODO expand on the documentation, add reference
class ColemanLiDescent(
    AbstractDescent[
        Y, Iterate.Primal, FunctionInfo.EvalGradHessian, _ColemanLiDescentState
    ],
    strict=True,
):
    """Coleman-Li descent."""

    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def init(self, y: Y, f_info_struct: FunctionInfo) -> _ColemanLiDescentState:  # pyright: ignore
        del f_info_struct
        # Dummy values of the right shape; unused.
        return _ColemanLiDescentState(y, RESULTS.successful)

    def query(  # pyright: ignore
        self,
        y: Y,
        f_info: FunctionInfo.EvalGradHessian,
        state: _ColemanLiDescentState,
    ) -> _ColemanLiDescentState:
        del state
        newton, result = _interior_reflected_newton_step(y, f_info, self.linear_solver)
        return _ColemanLiDescentState(newton, result)

    def step(  # pyright: ignore TODO
        self, step_size: Scalar, state: _ColemanLiDescentState
    ) -> tuple[Y, RESULTS]:
        return (-step_size * state.newton**ω).ω, state.result


ColemanLiDescent.__init__.__doc__ = """**Arguments**:

- `linear_solver`: The linear solver to use when computing the Newton step.
"""


# TODO: maybe change the name of this solver, since only the descent sticks fairly close
# to the method proposed by Coleman and Li. In particular, this solver is not a true
# trust-region method. Writing it as intended would entail computing a trust-region
# radius in the transformed space, which means that we need to either do the transform
# twice in two different places, or write a solver that applies it before handing over
# to the search. The function value itself is not transformed, and `y` would again be
# updated in the untransformed space. However, the main feature is the affine scaling,
# which angles the search direction away from the bounds. And this is what we do here.
# (I'm unconvinced that the trust region radius computed in the transformed space would
# be dramatically different from the one computed in the original space, but willing to
# convince myself otherwise with pen and paper.)
# Reference for this solver: https://link.springer.com/article/10.1007/BF01582221
class ColemanLi(AbstractOldBFGS[Y, Aux, FunctionInfo.EvalGradHessian], strict=True):
    """Coleman-Li solver for bounded optimisation problems. This solver uses the
    distance to the bounds to define a scaling operator for the Hessian and gradient
    before computing the direction in the transformed space. This results in a search
    direction that is angled away from the bounds.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: ColemanLiDescent
    search: BacktrackingArmijo
    verbose: frozenset[str]

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
        self.descent = ColemanLiDescent(linear_solver=lx.SVD())
        self.search = BacktrackingArmijo()
        self.verbose = verbose
