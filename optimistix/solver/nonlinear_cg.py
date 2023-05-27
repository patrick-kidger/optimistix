import functools as ft
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar

from ..line_search import AbstractDescent, OneDimensionalFunction
from ..linear_operator import AbstractLinearOperator, IdentityLinearOperator
from ..minimise import AbstractMinimiser, minimise, MinimiseProblem
from ..misc import max_norm, tree_full, tree_zeros, tree_zeros_like
from ..solution import RESULTS
from .nonlinear_cg_descent import hestenes_stiefel, NonlinearCGDescent


def _small(diffsize: Scalar) -> Bool[ArrayLike, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[ArrayLike, " "]:
    return jnp.invert(jnp.isfinite(rate))


def _converged(factor: Scalar, tol: float) -> Bool[ArrayLike, " "]:
    return (factor > 0) & (factor < tol)


class GradOnlyState(eqx.Module):
    descent_state: PyTree
    vector: PyTree[Array]
    operator: AbstractLinearOperator
    diff: PyTree[Array]
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    f_val: PyTree[Array]
    next_init: Array
    aux: Any
    step: Scalar


# note that this is called "GradOnly" and not "VecOnly," despite us often referring
# to the gradient and residual vectors by the name `vector`.
# This is because it doesn't make sense to use this in the least squares setting
# where `vector` is the residual vector, so we know we are always dealing with
# gradients.
class AbstractGradOnly(AbstractMinimiser):
    rtol: float
    atol: float
    line_search: AbstractMinimiser
    descent: AbstractDescent
    norm: Callable
    converged_tol: float

    def __init__(
        self,
        rtol: float,
        atol: float,
        line_search: AbstractMinimiser,
        descent: AbstractDescent,
        norm: Callable = max_norm,
        converged_tol: float = 1e-2,
    ):
        self.rtol = rtol
        self.atol = atol
        self.line_search = line_search
        self.descent = descent
        self.norm = norm
        self.converged_tol = converged_tol

    def init(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ):
        f0 = tree_full(f_struct, jnp.inf)
        aux = tree_zeros(aux_struct)
        y_zeros = tree_zeros_like(y)
        operator = IdentityLinearOperator(jax.eval_shape(lambda: y))
        descent_state = self.descent.init_state(
            problem, y, y_zeros, operator, None, args, {}
        )
        return GradOnlyState(
            descent_state=descent_state,
            vector=y_zeros,
            operator=operator,
            diff=jtu.tree_map(lambda x: jnp.full(x.shape, jnp.inf), y),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=jnp.array(RESULTS.successful),
            f_val=f0,
            next_init=jnp.array(1.0),
            aux=aux,
            step=jnp.array(0),
        )

    def step(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: GradOnlyState,
    ):
        def main_pass(y, state):
            descent = eqx.Partial(
                self.descent,
                descent_state=state.descent_state,
                args=args,
                options=options,
            )
            problem_1d = MinimiseProblem(
                OneDimensionalFunction(problem.fn, descent, y), has_aux=True
            )
            line_search_options = {
                "f0": state.f_val,
                "compute_f0": (state.step == 0),
                "vector": state.vector,
                "operator": state.operator,
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
                y0=init,
                args=args,
                options=line_search_options,
                max_steps=100,
            )
            (f_val, diff, new_aux, _, next_init) = line_sol.aux
            return f_val, diff, new_aux, line_sol.result, next_init

        def first_pass(y, state):
            return (
                jnp.inf,
                ω(y).call(jnp.zeros_like).ω,
                state.aux,
                RESULTS.successful,
                state.next_init,
            )

        # this lax.cond allows us to avoid an extra compilation of f(y) in the init.
        f_val, diff, new_aux, result, next_init = lax.cond(
            state.step == 0, first_pass, main_pass, y, state
        )
        new_y = (y**ω + diff**ω).ω
        new_grad, _ = jax.jacrev(problem.fn, has_aux=True)(new_y, args)
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(diff) / ω(scale)).ω)
        descent_state = self.descent.update_state(
            state.descent_state, diff, new_grad, state.operator, None, {}
        )
        result = jnp.where(
            result == RESULTS.max_steps_reached, RESULTS.successful, result
        )
        new_state = GradOnlyState(
            descent_state=descent_state,
            vector=new_grad,
            operator=state.operator,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=result,
            f_val=f_val,
            next_init=next_init,
            aux=new_aux,
            step=state.step + 1,
        )
        return new_y, new_state, new_aux

    def terminate(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: GradOnlyState,
    ):
        at_least_two = state.step > 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.converged_tol)
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (at_least_two & (small | diverged | converged))
        result = jnp.where(diverged, RESULTS.nonlinear_divergence, RESULTS.successful)
        result = jnp.where(linsolve_fail, state.result, result)
        return terminate, result

    def buffers(self, state: GradOnlyState):
        return ()


class GradOnly(AbstractGradOnly):
    pass


class NonlinearCG(AbstractGradOnly):
    def __init__(
        self,
        rtol: float,
        atol: float,
        line_search: AbstractMinimiser,
        norm: Callable = max_norm,
        converged_tol: float = 1e-2,
        method: Callable = hestenes_stiefel,
    ):
        self.rtol = rtol
        self.atol = atol
        self.line_search = line_search
        self.norm = norm
        self.converged_tol = converged_tol
        self.descent = NonlinearCGDescent(method)
