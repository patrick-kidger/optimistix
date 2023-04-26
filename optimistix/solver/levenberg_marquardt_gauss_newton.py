from typing import Any, Callable, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from ..custom_types import Scalar
from ..line_search import AbstractDescent, OneDimFunction
from ..linear_operator import AbstractLinearOperator
from ..minimise import AbstractMinimiser, minimise, MinimiseProblem
from ..misc import max_norm
from ..solution import RESULTS
from .backtracking import BacktrackingArmijo
from .iterative_dual import DirectIterativeDual, IndirectIterativeDual
from .misc import compute_jac_residual
from .newton_chord import Newton
from .quasi_newton import AbstractQuasiNewton
from .trust_region import ClassicalTrustRegion


class GNState(eqx.Module):
    step: Scalar
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    vector: Optional[PyTree[ArrayLike]]
    operator: Optional[AbstractLinearOperator]
    aux: Any
    f_i: Float[ArrayLike, " "]


class AbstractGaussNewton(AbstractQuasiNewton):
    atol: float
    rtol: float
    norm: Callable

    def init(self, problem, y, args, options):

        vector, operator, aux = compute_jac_residual(problem, y, args)
        #
        # WARNING: the `f_i` term in state is a footgun. We
        # should have something like `search_state` instead. However, to initialize
        # `search_state` we need to have `problem_1d` as in `self.step`. This requires
        # the descent function to get state as an argument. Maybe we create something
        # like `InitState(step, diffsize, ...)` without `search_state`, use this to
        # create `problem_1d`, and then use `problem_1d` to initialize `search_state`?
        # search state could then be passed via `QNState`. Seems a bit hacky though.
        #
        f_0 = problem.fn(y, args)

        return GNState(
            step=jnp.array(0),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=jnp.array(RESULTS.successful),
            vector=vector,
            operator=operator,
            aux=aux,
            f_i=f_0,
        )

    def step(self, problem, y, args, options, state):
        aux = state.aux
        descent = eqxi.Partial(
            self.descent,
            problem=problem,
            y=y,
            args=args,
            state=state,
            options=options,
            vector=state.vector,
            operator=state.operator,
        )

        problem_1d = MinimiseProblem(
            OneDimFunction(problem.fn, descent, y), has_aux=True
        )
        options["vector"] = state.vector
        options["operator"] = state.operator
        options["f_0"] = state.f_i

        line_sol = minimise(
            problem_1d,
            self.line_search,
            jnp.array(1.0),
            args=args,
            options=options,
            max_steps=512,
        )

        (diff, new_aux) = line_sol.aux
        new_y = (ω(y) + ω(diff)).ω
        vector, operator, _ = compute_jac_residual(problem, new_y, options, args)
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(diff) / ω(scale)).ω)
        new_state = GNState(
            step=state.step + 1,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=line_sol.result,
            vector=vector,
            operator=operator,
            aux=new_aux,
            f_i=line_sol.state.f_prev,
        )
        return new_y, new_state, aux

    def buffers(self, state):
        return ()


class LevenbergMarquardt(AbstractGaussNewton):
    line_search: AbstractMinimiser
    descent: AbstractDescent
    converged_tol: float

    def __init__(
        self,
        atol: float,
        rtol: float,
        norm=max_norm,
        converged_tol: float = 1e-2,
        lambda_0: Float[ArrayLike, ""] = 1e-3,
    ):
        # WARNING: atol and rtol are being used both for
        # IndirectIterativeDual and for self! This may be bad
        # practice.
        self.atol = atol
        self.rtol = rtol
        self.norm = norm
        self.converged_tol = converged_tol
        self.line_search = ClassicalTrustRegion()
        self.descent = IndirectIterativeDual(
            gauss_newton=True,
            lambda_0=lambda_0,
            root_finder=Newton(rtol, atol, lower=1e-6),
        )


class DirectLevenbergMarquardt(AbstractGaussNewton):
    converged_tol: float

    def __init__(
        self,
        atol: float,
        rtol: float,
        norm=max_norm,
        converged_tol: float = 1e-2,
        backtrack_slope: float = 0.1,
        decrease_factor: float = 0.5,
    ):
        self.atol = atol
        self.rtol = rtol
        self.norm = norm
        self.converged_tol = converged_tol
        self.descent = DirectIterativeDual(gauss_newton=True)
        # TODO(raderj): should use a slightly different backtracking algo
        self.line_search = BacktrackingArmijo(
            backtrack_slope=backtrack_slope, decrease_factor=decrease_factor
        )
