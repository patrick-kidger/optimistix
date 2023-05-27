from typing import Any, Callable, cast, Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, Int, PyTree, Scalar

from ..custom_types import sentinel
from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractDescent
from ..misc import tree_where, two_norm
from ..root_find import AbstractRootFinder, root_find, RootFindProblem
from ..solution import RESULTS
from .misc import quadratic_predicted_reduction


#
# NOTE: This method is usually called Levenberg-Marquardt. However,
# Levenberg-Marquard often refers specifically to the case where this approach
# is applied in the Gauss-Newton setting. For this reason, we refer to the approach
# by the more generic name "iterative dual."
#
# Iterative dual is a method of solving for the descent direction given a
# trust region radius. It does this by solving the dual problem
# `(B + lambda I) p = r` for `p`, where `B` is the quasi-Newton matrix,
# lambda is the dual parameter (the dual parameterisation of the
# trust region radius), `I` is the identity, and `r` is the vector of residuals.
#
# Iterative dual is approached in one of two ways:
# 1. set the trust region radius and find the Levenberg-Marquadt parameter
# `lambda` which will give approximately solve the trust region problem. ie.
# solve the dual of the trust region problem.
#
# 2. set the Levenberg-Marquadt parameter `lambda` directly.
#
# Respectively, this is the indirect and direct approach to iterative dual.
# The direct approach is common in practice, and is often interpreted as
# interpolating between quasi-Newton and gradient based approaches.
#
# The indirect approach is very interpretable in the classical trust region sense.
# Note however, if `B` is the quasi-Newton Hessian approximation and `g` the
# gradient, that `||p(lambda)||` is dependent upon `B`. Specifically, decompose
# `B = QLQ^T` via the spectral decomposition with eigenvectors $q_i$ and corresponding
# eigenvalues $l_i$. Then
# ```
# ||p(lambda)||^2 = sum(((q_j^T g)^2)/(l_1 + lambda)^2)
# ```
# The consequence of this is that the relationship between lambda and the trust region
# radius changes each iteration as `B` is updated. For this reason, many researchers
# prefer the more interpretable indirect approach. This is what Moré used in their
# classical implementation of the algorithm as well (see Moré, "The Levenberg-Marquardt
# Algorithm: Implementation and Theory.")
#
def _small(diffsize: Scalar) -> Bool[Array, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[Array, " "]:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: float) -> Bool[Array, " "]:
    return (factor > 0) & (factor < tol)


class _IndirectDualState(eqx.Module):
    delta: Scalar
    diffsize: Scalar
    diffsize_prev: Scalar
    lower_bound: Scalar
    upper_bound: Scalar
    result: RESULTS
    step: Int[Array, ""]


class _IndirectDualRootFind(AbstractRootFinder):
    gauss_newton: bool
    converged_tol: float

    def init(
        self,
        problem: RootFindProblem,
        y: Array,
        args: Any,
        options: dict[str, Any],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ):
        del aux_struct, f_struct
        try:
            delta = options["delta"]
        except KeyError:
            raise ValueError(
                "The indirect iterative dual root find needs delta "
                "(trust region radius) passed via `options['delta']`"
            )

        if self.gauss_newton:
            try:
                vector = options["vector"]
                operator = options["operator"]
            except KeyError:
                raise ValueError(
                    "The indirect iterative dual root find with "
                    "`gauss_newton=True` needs the operator and vector passed via "
                    "`options['operator']` and `options['vector']`."
                )
            grad = operator.transpose().mv(vector)
        else:
            try:
                grad = options["vector"]
            except KeyError:
                raise ValueError(
                    "The indirect iterative dual root find with "
                    "`gauss_newton=False` needs the vector passed via "
                    "`options['vector']`."
                )
        delta_nonzero = delta > jnp.finfo(delta.dtype).eps
        safe_delta = jnp.where(delta_nonzero, delta, 1)
        upper_bound = jnp.where(delta_nonzero, two_norm(grad) / safe_delta, jnp.inf)
        return _IndirectDualState(
            delta=delta,
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            lower_bound=jnp.array(0.0),
            upper_bound=upper_bound,
            result=jnp.array(RESULTS.successful),
            step=jnp.array(0),
        )

    def step(
        self,
        problem: RootFindProblem,
        y: Array,
        args: Any,
        options: dict[str, Any],
        state: _IndirectDualState,
    ):
        # avoid an extra compilation of problem.fn in the init.
        _y = jnp.where(state.step == 0, 0, y)
        lambda_in_bounds = (state.lower_bound < y) & (y < state.upper_bound)
        new_y = jnp.where(
            lambda_in_bounds,
            _y,
            jnp.maximum(
                1e-3 * state.upper_bound,
                jnp.sqrt(state.upper_bound * state.lower_bound),
            ),
        )
        # TODO(raderj): track down a link to reference for this.
        f_val, aux = problem.fn(_y, args)
        f_grad, _ = jax.grad(problem.fn, has_aux=True)(_y, args)
        grad_nonzero = f_grad < jnp.finfo(f_grad.dtype).eps
        safe_grad = jnp.where(grad_nonzero, f_grad, 1)
        factor = cast(
            Array, jnp.where(grad_nonzero, f_val / safe_grad, jnp.array(jnp.inf))
        )
        upper_bound = jnp.where(f_val < 0, y, state.upper_bound)
        lower_bound = jnp.maximum(state.lower_bound, y - factor)
        diff = -((f_val - state.delta) / state.delta) * factor
        new_y = y + diff
        new_y = jnp.where(state.step == 0, y, new_y)
        upper_bound = jnp.where(state.step == 0, state.upper_bound, upper_bound)
        lower_bound = jnp.where((state.step == 0) & grad_nonzero, -factor, lower_bound)
        new_state = _IndirectDualState(
            state.delta,
            diff,
            state.diffsize,
            lower_bound,
            upper_bound,
            jnp.array(RESULTS.successful),
            state.step + 1,
        )
        return jnp.clip(new_y, a_min=0), new_state, aux

    def terminate(
        self,
        problem: RootFindProblem,
        y: Array,
        args: Any,
        options: dict[str, Any],
        state: _IndirectDualState,
    ):
        del problem, args, options
        y_zero = jnp.abs(y) < jnp.finfo(y.dtype).eps
        at_least_two = state.step >= 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.converged_tol)
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (at_least_two & (small | diverged | converged))
        in_bounds = (y > state.lower_bound) & (y < state.upper_bound)
        terminate = terminate & in_bounds
        terminate = terminate | y_zero
        result = jnp.where(diverged, RESULTS.nonlinear_divergence, RESULTS.successful)
        result = jnp.where(y_zero, RESULTS.successful, result)
        result = jnp.where(linsolve_fail, state.result, result)
        return terminate, result

    def buffers(self, state: _IndirectDualState):
        return ()


class IterativeDualState(eqx.Module):
    y: PyTree[Array]
    problem: AbstractIterativeProblem
    vector: PyTree[Array]
    operator: lx.AbstractLinearOperator


class _Damped(eqx.Module):
    fn: Callable
    damping: Float[Array, " "]

    def __call__(self, y, args):
        damping = jnp.sqrt(self.damping)
        f, aux = self.fn(y, args)
        damped = jtu.tree_map(lambda yi: damping * yi, y)
        return (f, damped), aux


class _DirectIterativeDual(AbstractDescent[IterativeDualState]):
    gauss_newton: bool
    modify_jac: Callable[
        [lx.JacobianLinearOperator], lx.AbstractLinearOperator
    ] = lx.linearise

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator] = None,
        operator_inv: Optional[lx.AbstractLinearOperator] = None,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        if operator is None:
            assert False
        return IterativeDualState(y, problem, vector, operator)

    def update_state(
        self,
        descent_state: IterativeDualState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: Optional[dict[str, Any]] = None,
    ):
        return IterativeDualState(
            descent_state.y, descent_state.problem, vector, operator
        )

    def __call__(
        self,
        delta: Scalar,
        descent_state: IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ):
        if self.gauss_newton:
            vector = (descent_state.vector, ω(descent_state.y).call(jnp.zeros_like).ω)
            operator = lx.JacobianLinearOperator(
                _Damped(descent_state.problem.fn, delta),
                descent_state.y,
                args,
                _has_aux=True,
            )
        else:
            vector = descent_state.vector
            operator = descent_state.operator + delta * lx.IdentityLinearOperator(
                descent_state.operator.in_structure()
            )
        operator = self.modify_jac(operator)
        linear_soln = lx.linear_solve(operator, vector, lx.QR(), throw=False)
        diff = (-linear_soln.value**ω).ω
        return diff, linear_soln.result

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ):
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )


class DirectIterativeDual(_DirectIterativeDual):
    def __call__(
        self,
        delta: Scalar,
        descent_state: IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ):
        if descent_state.operator is None:
            raise ValueError(
                "`operator` must be passed to `DirectIterativeDual`. "
                "Note that `operator_inv` is not currently supported for this descent."
            )

        delta_nonzero = delta > jnp.finfo(delta.dtype).eps
        if self.gauss_newton:
            vector = (descent_state.vector, ω(descent_state.y).call(jnp.zeros_like).ω)
            operator = lx.JacobianLinearOperator(
                _Damped(
                    descent_state.problem.fn,
                    jnp.where(delta_nonzero, 1 / delta, jnp.inf),
                ),
                descent_state.y,
                args,
                _has_aux=True,
            )
        else:
            vector = descent_state.vector
            operator = descent_state.operator + jnp.where(
                delta_nonzero, 1 / delta, jnp.inf
            ) * lx.IdentityLinearOperator(descent_state.operator.in_structure())
        operator = self.modify_jac(operator)
        linear_soln = lx.linear_solve(operator, vector, lx.QR(), throw=False)
        no_diff = jtu.tree_map(jnp.zeros_like, linear_soln.value)
        diff = tree_where(delta_nonzero, (-linear_soln.value**ω).ω, no_diff)
        return diff, linear_soln.result


class IndirectIterativeDual(AbstractDescent[IterativeDualState]):

    #
    # Indirect iterative dual finds the `λ` to match the
    # trust region radius by applying Newton's root finding method to
    # `φ(p) = ||p(λ)|| - δ`
    # where `δ` is the trust region radius.
    #
    # Moré found a clever way to compute `dφ/dλ = -||q||^2/||p(λ)||` where `q` is
    # defined as: `q = R^(-1) p`, for `R` as in the QR decomposition of
    # `(B + λ I)`.
    #
    # TODO(raderj): write a solver in root_finder which specifically assumes iterative
    # dual so we can use the trick (or at least see if it's worth doing.)
    # TODO(raderj): Use Householder + Givens method.
    #
    gauss_newton: bool
    lambda_0: Float[Array, ""]
    root_finder: AbstractRootFinder
    solver: lx.AbstractLinearSolver
    tr_reg: Optional[lx.PyTreeLinearOperator]
    norm: Callable
    modify_jac: Callable[[lx.JacobianLinearOperator], lx.AbstractLinearOperator]

    def __init__(
        self,
        gauss_newton: bool,
        lambda_0: float,
        root_finder: AbstractRootFinder = sentinel,
        solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False),
        tr_reg: Optional[lx.PyTreeLinearOperator] = None,
        norm: Callable = two_norm,
        modify_jac: Callable[
            [lx.JacobianLinearOperator], lx.AbstractLinearOperator
        ] = lx.linearise,
    ):
        self.gauss_newton = gauss_newton
        self.lambda_0 = jnp.array(lambda_0)
        if root_finder is sentinel:
            self.root_finder = _IndirectDualRootFind(self.gauss_newton, 1e-3)
        else:
            self.root_finder = root_finder
        self.solver = solver
        self.tr_reg = tr_reg
        self.norm = norm
        self.modify_jac = modify_jac

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        if operator is None:
            assert False
        return IterativeDualState(y, problem, vector, operator)

    def update_state(
        self,
        descent_state: IterativeDualState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: Optional[dict[str, Any]] = None,
    ):
        return IterativeDualState(
            descent_state.y, descent_state.problem, vector, operator
        )

    def __call__(
        self,
        delta: Scalar,
        descent_state: IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ):
        if descent_state.operator is None:
            raise ValueError(
                "`operator` must be passed to "
                " `IndirectDirectIterativeDual`. Note that `operator_inv` is "
                "not currently supported for this descent."
            )

        direct_dual = eqx.Partial(
            _DirectIterativeDual(self.gauss_newton, self.modify_jac),
            descent_state=descent_state,
            args=args,
            options=options,
        )
        newton_soln = lx.linear_solve(
            descent_state.operator, (-descent_state.vector**ω).ω, self.solver
        )
        # NOTE: try delta = delta * self.norm(newton_step).
        # this scales the trust and sets the natural bound `delta = 1`.
        newton_step = (-ω(newton_soln.value)).ω
        newton_result = newton_soln.result
        tr_reg = self.tr_reg

        if tr_reg is None:
            tr_reg = lx.IdentityLinearOperator(jax.eval_shape(lambda: newton_step))

        def comparison_fn(
            lambda_i: Scalar,
            args: Any,
        ):
            (step, _) = direct_dual(lambda_i)
            step_norm = self.norm(step)
            return step_norm - delta

        def accept_newton():
            return newton_step, newton_result

        def reject_newton():
            root_find_problem = RootFindProblem(comparison_fn, has_aux=False)
            root_find_options = {
                "vector": descent_state.vector,
                "operator": descent_state.operator,
                "delta": delta,
            }
            # I don't love the max_steps here.
            lambda_out = root_find(
                root_find_problem,
                self.root_finder,
                self.lambda_0,
                args,
                root_find_options,
                max_steps=32,
                throw=False,
            ).value
            return direct_dual(lambda_out)

        newton_norm = self.norm(newton_step)
        return lax.cond(newton_norm < delta, accept_newton, reject_newton)

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ):
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )
