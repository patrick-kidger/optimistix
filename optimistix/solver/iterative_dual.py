from typing import Any, Callable, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, Bool, Float, Int, PyTree, Scalar

from ..custom_types import sentinel
from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractProxyDescent
from ..linear_operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    JacobianLinearOperator,
    linearise,
    PyTreeLinearOperator,
)
from ..linear_solve import AbstractLinearSolver, AutoLinearSolver, linear_solve
from ..misc import tree_inner_prod, tree_where, two_norm
from ..root_find import AbstractRootFinder, root_find, RootFindProblem
from ..solution import RESULTS
from .qr import QR


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


def _converged(factor: Scalar, tol: Scalar) -> Bool[Array, " "]:
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
        self, problem: RootFindProblem, y: Array, args: Any, options: dict[str, Any]
    ):
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

        upper_bound = two_norm(grad) / delta
        return _IndirectDualState(
            delta=delta,
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            lower_bound=jnp.array(-jnp.inf),
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
        f_val, aux = problem.fn(_y, args)
        f_grad, _ = jax.jacfwd(problem.fn)(_y, args)
        factor = f_val / f_grad
        upper_bound = jnp.where(f_val < 0, y, state.upper_bound)
        lower_bound = jnp.maximum(state.lower_bound, y - factor)
        diff = -((f_val - state.delta) / state.delta) * factor
        new_y = y + diff
        new_y = jnp.where(state.step == 0, y, new_y)
        upper_bound = jnp.where(state.step == 0, state.upper_bound, upper_bound)
        # this case handling is to avoid an extra compilation of problem.fn in the init
        # to compute the first lower bound.
        lower_bound = jnp.where(state.step == 0, -factor, lower_bound)
        new_state = _IndirectDualState(
            state.delta,
            diff,
            state.diffsize,
            lower_bound,
            upper_bound,
            jnp.array(RESULTS.successful),
            state.step + 1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        problem: RootFindProblem,
        y: Array,
        args: Any,
        options: dict[str, Any],
        state: _IndirectDualState,
    ):
        del problem, y, args, options
        at_least_two = state.step >= 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.converged_tol)
        linsolve_fail = state.result != RESULTS.successful  # pyright: ignore
        terminate = linsolve_fail | (at_least_two & (small | diverged | converged))
        result = jnp.where(
            diverged, RESULTS.nonlinear_divergence, RESULTS.successful
        )  # pyright: ignore
        result = jnp.where(linsolve_fail, state.result, result)  # pyright: ignore
        return terminate, result

    def buffers(self, state: _IndirectDualState):
        return ()


class IterativeDualState(eqx.Module):
    y: PyTree[Array]
    problem: AbstractIterativeProblem
    vector: PyTree[Array]
    operator: AbstractLinearOperator


class _Damped(eqx.Module):
    fn: Callable
    damping: Float[Array, " "]

    def __call__(self, y, args):
        damping = jnp.sqrt(self.damping)
        f, aux = self.fn(y, args)
        damped = jtu.tree_map(lambda yi: damping * yi, y)
        return (f, damped), aux


def _predicted_reduction(
    gauss_newton: bool,
    diff: PyTree[Array],
    descent_state: IterativeDualState,
    args: Any,
    options: dict[str, Any],
):
    # The predicted reduction of the iterative dual. This is the model quadratic
    # model function of classical trust region methods localized around f(x).
    # ie. `m(p) = g^t p + 1/2 p^T B p` where `g` is the gradient, `B` the
    # Quasi-Newton approximation to the Hessian, and `p` the
    # descent direction (diff).
    #
    # in the Gauss-Newton setting we compute
    # ```0.5 * [(Jp - r)^T (Jp - r) - r^T r]```
    # which is equivalent when `B = J^T J` and `g = J^T r`.
    if gauss_newton:
        rtr = two_norm(descent_state.vector) ** 2
        jacobian_term = (
            two_norm((ω(descent_state.operator.mv(diff)) - ω(descent_state.vector)).ω)
            ** 2
        )
        return 0.5 * (jacobian_term - rtr)
    else:
        operator_quadratic = 0.5 * tree_inner_prod(
            diff, descent_state.operator.mv(diff)
        )
        steepest_descent = tree_inner_prod(descent_state.vector, diff)
        return (operator_quadratic**ω + steepest_descent**ω).ω


class _DirectIterativeDual(AbstractProxyDescent[IterativeDualState]):
    gauss_newton: bool
    modify_jac: Callable[[JacobianLinearOperator], AbstractLinearOperator] = linearise

    def init_state(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return IterativeDualState(y, problem, vector, operator)

    def update_state(
        self,
        descent_state: IterativeDualState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
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
        delta_nonzero = delta > jnp.finfo(delta.dtype).eps
        if self.gauss_newton:
            vector = (descent_state.vector, ω(descent_state.y).call(jnp.zeros_like).ω)
            operator = JacobianLinearOperator(
                _Damped(descent_state.problem.fn, jnp.where(delta_nonzero, delta, 0)),
                descent_state.y,
                args,
                _has_aux=True,
            )
        else:
            vector = descent_state.vector
            operator = descent_state.operator + jnp.where(
                delta_nonzero, delta, 0
            ) * IdentityLinearOperator(descent_state.operator.in_structure())
        operator = self.modify_jac(operator)
        eqxi.error_if(
            delta, delta < 1e-6, "The dual (LM) parameter must be >1e-6 to ensure psd"
        )
        linear_soln = linear_solve(operator, vector, QR(), throw=False)
        diff = (-linear_soln.value**ω).ω
        return diff, linear_soln.result

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: IterativeDualState,
        args: Any,
        options: dict[str, Any],
    ):
        return _predicted_reduction(
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

        delta_nonzero = delta > jnp.finfo(delta.dtype).eps
        if self.gauss_newton:
            vector = (descent_state.vector, ω(descent_state.y).call(jnp.zeros_like).ω)
            operator = JacobianLinearOperator(
                _Damped(
                    descent_state.problem.fn,
                    jnp.where(delta_nonzero, 1 / delta, jnp.inf),
                ),
                descent_state.y,
                args,
                _has_aux=True,
            )
        else:
            # WARNING: this branch is yet untested.
            vector = descent_state.vector
            operator = descent_state.operator + jnp.where(
                delta_nonzero, 1 / delta, jnp.inf
            ) * IdentityLinearOperator(descent_state.operator.in_structure())
        operator = self.modify_jac(operator)
        eqxi.error_if(
            delta, delta < 1e-6, "The dual (LM) parameter must be >1e-6 to ensure psd"
        )
        linear_soln = linear_solve(operator, vector, QR(), throw=False)
        no_diff = jtu.tree_map(jnp.zeros_like, linear_soln.value)
        diff = tree_where(delta_nonzero, (-linear_soln.value**ω).ω, no_diff)
        return diff, linear_soln.result


class IndirectIterativeDual(AbstractProxyDescent[IterativeDualState]):

    #
    # Indirect iterative dual finds the `lambda` to match the
    # trust region radius by applying Newton's root finding method to
    # `phi.alt = 1/||p(lambda)|| - 1/delta`
    # As reccomended by [Hebden 1973 p. 8] and later [Nocedal Wright].
    # We could also use
    # `phi(lambda) = ||p(lambda)|| - delta`
    # but found it less numerically stable
    #
    # Moré found a clever way to compute `phi' = -||q||^2/||p(lambda)||` where `q` is
    # defined as: `q = R^(-1) p`, for `R` as in the QR decomposition of
    # `(B + lambda I)`. This can similarly be applied to the phi in Hebden with
    # `phi' = ||q||^2/||p(lambda)||^(3/2)`
    #
    # TODO(raderj): write a solver in root_finder which specifically assumes iterative
    # dual so we can use the trick (or at least see if it's worth doing.)
    # TODO(raderj): Use Householder + Givens method.
    #
    gauss_newton: bool
    lambda_0: Float[Array, ""]
    root_finder: AbstractRootFinder
    solver: AbstractLinearSolver
    tr_reg: Optional[PyTreeLinearOperator]
    norm: Callable
    modify_jac: Callable[[JacobianLinearOperator], AbstractLinearOperator]

    def __init__(
        self,
        gauss_newton: bool,
        lambda_0: float,
        root_finder: AbstractRootFinder = sentinel,
        solver: AbstractLinearSolver = AutoLinearSolver(well_posed=False),
        tr_reg: Optional[PyTreeLinearOperator] = None,
        norm: Callable = two_norm,
        modify_jac: Callable[
            [JacobianLinearOperator], AbstractLinearOperator
        ] = linearise,
    ):
        self.gauss_newton = gauss_newton
        self.lambda_0 = jnp.array(lambda_0)
        if root_finder == sentinel:
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
        operator: AbstractLinearOperator,
        args: Optional[Any] = None,
        options: Optional[dict[str, Any]] = {},
    ):
        return IterativeDualState(y, problem, vector, operator)

    def update_state(
        self,
        descent_state: IterativeDualState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: AbstractLinearOperator,
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
        direct_dual = eqx.Partial(
            _DirectIterativeDual(self.gauss_newton, self.modify_jac),
            descent_state=descent_state,
            args=args,
            options=options,
        )
        newton_soln = linear_solve(
            descent_state.operator, (-descent_state.vector**ω).ω, self.solver
        )
        # NOTE: try delta = delta * self.norm(newton_step).
        # this scales the trust and sets the natural bound `delta = 1`.
        newton_step = (-ω(newton_soln.value)).ω
        newton_result = newton_soln.result
        tr_reg = self.tr_reg

        if tr_reg is None:
            tr_reg = IdentityLinearOperator(jax.eval_shape(lambda: newton_step))

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
        return _predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )
