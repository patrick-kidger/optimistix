from typing import Callable

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import ArrayLike, Float

from ..line_search import AbstractDescent, AbstractLineSearch, OneDimProblem
from ..minimise import minimise, MinimiseProblem
from ..misc import max_norm
from ..solution import RESULTS
from .backtracking import BacktrackingArmijo
from .iterative_dual import DirectIterativeDual, IndirectIterativeDual
from .misc import compute_jac_residual
from .newton_chord import Newton
from .quasi_newton import AbstractQuasiNewton, QNState
from .trust_region import ClassicalTrustRegion


class AbstractGaussNewton(AbstractQuasiNewton):
    atol: float
    rtol: float
    line_search: AbstractLineSearch
    descent: AbstractDescent
    norm: Callable

    def init(self, problem, y, args, options):

        vector, operator, aux = compute_jac_residual(problem, y, options, args)
        #
        # WARNING: there are some footguns throughout regarding this f_i! Really we
        # should have something like search_state and use this. However, to initialize
        # search_state we need to have problem_1d as in the step, which requires
        # the descent function to get state as an argument. Maybe we create something
        # like InitState(step, diffsize, ...) with no search_state, use this to
        # create problem_1d, and then use problem_1d to initialize search_state.
        # search state can then be passed via QNState.
        #
        f_0 = problem.fn(y, args)

        return QNState(
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
        # WARNING: Literally the only difference between this and bfgs is
        # the init. All the state updating happens in update_state.
        # should we
        # a) remove update state and toss it into step.
        # b) write step in quasi-Newton and then just implement update_state?
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
            OneDimProblem(problem.fn, descent, y), has_aux=True
        )
        options["vector"] = state.vector
        jax.debug.print("residual_vec: {}", state.vector)
        jax.debug.print("y: {}", y)
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
        jax.debug.print("line_sol returned stuff: {}", line_sol.value)

        (diff, new_aux) = line_sol.aux
        new_y = (ω(y) + ω(diff)).ω
        vector, operator, _ = compute_jac_residual(problem, new_y, options, args)
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(diff) / ω(scale)).ω)
        new_state = QNState(
            step=state.step + 1,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=line_sol.result,
            vector=vector,
            operator=operator,
            aux=new_aux,
            # BAD!
            f_i=line_sol.state.f_prev,
        )
        jax.debug.print("new_y: {}", new_y)

        return new_y, new_state, aux

    def buffer(self, state):
        return ()


#
# Yep, LevenbergMarquardt is just an alias of GaussNewton with a default
# choice of descent and line search. This is similar to the popular implementation
# of Moré (1978) "The Levenberg-Marquardt Algorithm: Implementation and Theory."
#


class LevenbergMarquardt(AbstractGaussNewton):
    def __init__(
        self,
        atol: float,
        rtol: float,
        lambda_0: Float[ArrayLike, " "],
        converged_tol: float = 1e-2,
        norm=max_norm,
    ):
        # WARNING: atol and rtol are being used both for
        # IndirectIterativeDual and for self! This may be bad
        # practice.
        self.descent = IndirectIterativeDual(
            gauss_newton=True,
            lambda_0=lambda_0,
            root_finder=Newton(rtol, atol, lower=1e-10),
        )
        self.line_search = ClassicalTrustRegion()
        self.atol = atol
        self.rtol = rtol
        self.converged_tol = converged_tol
        self.norm = norm


class DirectLevenbergMarquardt(AbstractGaussNewton):
    # should use a slightly different backtracking algo
    descent = DirectIterativeDual(gauss_newton=True)
    line_search = BacktrackingArmijo(backtrack_slope=0.1, decrease_factor=0.5)
