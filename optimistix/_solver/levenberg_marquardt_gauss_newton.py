import functools as ft
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Float, Int, PyTree, Scalar

from .._least_squares import AbstractLeastSquaresSolver, LeastSquaresProblem
from .._line_search import AbstractDescent, OneDimensionalFunction
from .._minimise import AbstractMinimiser, minimise, MinimiseProblem
from .._misc import max_norm, tree_full_like
from .._solution import RESULTS
from .iterative_dual import DirectIterativeDual, IndirectIterativeDual
from .misc import compute_jac_residual, two_norm
from .trust_region import ClassicalTrustRegion


class GNState(eqx.Module):
    descent_state: PyTree
    vector: PyTree[ArrayLike]
    operator: lx.AbstractLinearOperator
    diff: PyTree[Array]
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    f_val: PyTree[Array]
    f_prev: PyTree[Array]
    next_init: Array
    aux: Any
    step: Int[Array, ""]


class AbstractGaussNewton(AbstractLeastSquaresSolver):
    rtol: float
    atol: float
    line_search: AbstractMinimiser
    descent: AbstractDescent
    norm: Callable
    converged_tol: float

    def init(
        self,
        problem: LeastSquaresProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
    ):
        del options, f_struct, aux_struct
        f0 = jnp.array(jnp.inf)
        vector, operator, aux = compute_jac_residual(problem, y, args)
        descent_state = self.descent.init_state(
            problem, y, vector, operator, None, args, {}
        )
        return GNState(
            descent_state=descent_state,
            vector=vector,
            operator=operator,
            diff=tree_full_like(y, jnp.inf),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=RESULTS.successful,
            f_val=f0,
            f_prev=f0,
            next_init=jnp.array(1.0),
            aux=aux,
            step=jnp.array(0),
        )

    def step(
        self,
        problem: LeastSquaresProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: GNState,
    ):
        descent = eqx.Partial(
            self.descent,
            descent_state=state.descent_state,
            args=args,
            options=options,
        )

        def line_search_problem(x, args):
            residual, aux = problem.fn(x, args)
            return two_norm(residual), aux

        problem_1d = MinimiseProblem(
            OneDimensionalFunction(line_search_problem, descent, y), has_aux=True
        )
        line_search_options = {
            "f0": state.f_val,
            "compute_f0": (state.step == 0),
            "vector": state.vector,
            "operator": state.operator,
            "diff": y,
        }

        line_search_options["predicted_reduction"] = ft.partial(
            self.descent.predicted_reduction,
            descent_state=state.descent_state,
            args=args,
            options={},
        )

        init = jnp.where(
            state.step == 0,
            self.line_search.first_init(state.vector, state.operator, options),
            state.next_init,
        )
        line_sol = minimise(
            problem_1d,
            self.line_search,
            init,
            args=args,
            options=line_search_options,
            max_steps=100,
            throw=False,
        )
        (f_val, diff, new_aux, _, next_init) = line_sol.aux
        new_y = (ω(y) + ω(diff)).ω
        vector, operator, _ = compute_jac_residual(problem, new_y, args)
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(diff) / ω(scale)).ω)
        descent_state = self.descent.update_state(
            state.descent_state, state.diff, vector, operator, None, options
        )
        new_state = GNState(
            descent_state=descent_state,
            vector=vector,
            operator=operator,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=RESULTS.successful,
            f_val=f_val,
            f_prev=state.f_val,
            next_init=next_init,
            aux=new_aux,
            step=state.step + 1,
        )
        # notice that this is state.aux, not new_state.aux or aux.
        # we assume aux is the aux at f(y), but line_search returns
        # aux at f(y_new), which is new_state.aux. So, we simply delay
        # the return of aux for one eval
        return new_y, new_state, state.aux

    def terminate(
        self,
        problem: LeastSquaresProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: GNState,
    ):
        at_least_two = state.step >= 2
        f_diff = jnp.abs(state.f_val - state.f_prev)
        converged = f_diff < self.rtol * jnp.abs(state.f_prev) + self.atol
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (at_least_two & converged)
        result = RESULTS.where(linsolve_fail, state.result, RESULTS.successful)
        return terminate, result

    def buffers(self, state: GNState):
        return ()


class GaussNewton(AbstractGaussNewton):
    rtol: float
    atol: float
    line_search: AbstractMinimiser
    descent: AbstractDescent
    norm: Callable = max_norm
    converged_tol: float = 1e-2


class IndirectLevenbergMarquardt(AbstractGaussNewton):
    line_search: AbstractMinimiser
    descent: AbstractDescent
    converged_tol: float

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm=max_norm,
        converged_tol: float = 1e-2,
        lambda_0: Float[ArrayLike, ""] = 1e-3,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.converged_tol = converged_tol
        self.line_search = ClassicalTrustRegion()
        self.descent = IndirectIterativeDual(
            gauss_newton=True,
            lambda_0=lambda_0,
        )


class LevenbergMarquardt(AbstractGaussNewton):
    converged_tol: float

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm=max_norm,
        converged_tol: float = 1e-2,
        backtrack_slope: float = 0.1,
        decrease_factor: float = 0.5,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.converged_tol = converged_tol
        self.descent = DirectIterativeDual(gauss_newton=True)
        self.line_search = ClassicalTrustRegion()
