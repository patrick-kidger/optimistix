import functools as ft
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree

from ..custom_types import Scalar
from ..line_search import AbstractDescent, AbstractProxyDescent, OneDimensionalFunction
from ..linear_operator import AbstractLinearOperator, PyTreeLinearOperator
from ..minimise import AbstractMinimiser, minimise, MinimiseProblem
from ..misc import max_norm, tree_inner_prod, two_norm
from ..solution import RESULTS
from .descent import UnnormalisedNewton
from .misc import compute_hess_grad


def _outer(tree):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf2: jnp.tensordot(x, leaf2, axes=0), tree)

    return jtu.tree_map(leaf_fn, tree)


def _small(diffsize: Scalar) -> Bool[ArrayLike, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[ArrayLike, " "]:
    return jnp.invert(jnp.isfinite(rate))


def _converged(factor: Scalar, tol: Scalar) -> Bool[ArrayLike, " "]:
    return (factor > 0) & (factor < tol)


class BFGSState(eqx.Module):
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


#
# Right now, this is performing the update `B_k -> B_(k + 1)` which solves
# `B_k p = g`. We could just as well update `Binv_k -> Binv_(k + 1)`, ie.
# `p = Binv_k g` is just the matrix vector product.
#
class BFGS(AbstractMinimiser):
    atol: float
    rtol: float
    line_search: AbstractMinimiser
    descent: AbstractDescent
    norm: Callable
    converged_tol: float
    use_inverse: bool

    def __init__(
        self,
        atol: float,
        rtol: float,
        line_search: AbstractMinimiser,
        descent: AbstractDescent = UnnormalisedNewton(),
        norm: Callable = max_norm,
        converged_tol: float = 1e-2,
        use_inverse: bool = True,
    ):

        self.atol = atol
        self.rtol = rtol
        self.line_search = line_search
        self.descent = descent
        self.norm = norm
        self.converged_tol = converged_tol
        self.use_inverse = use_inverse

    def init(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
    ):
        f0, aux = jtu.tree_map(
            lambda x: jnp.full(x.shape, jnp.inf), jax.eval_shape(problem.fn, y, args)
        )
        # TODO(raderj): remove this and replace with vector and Identity.
        # Note however the downstream we need operator.pytree, which the
        # IdentityLinearOperator does not have.
        vector, operator, aux = compute_hess_grad(problem, y, args)
        descent_state = self.descent.init_state(problem, y, vector, operator, args, {})

        return BFGSState(
            descent_state=descent_state,
            vector=vector,
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
        state: BFGSState,
    ):
        descent = eqx.Partial(
            self.descent,
            descent_state=state.descent_state,
            args=args,
            options=options,
        )
        problem_1d = MinimiseProblem(
            OneDimensionalFunction(problem, descent, y), has_aux=True
        )
        # TODO(raderj): pass f0
        line_search_options = {
            "f0": state.f_val,
            "compute_f0": (state.step == 0),
            "vector": state.vector,
            "operator": state.operator,
        }

        if isinstance(self.descent, AbstractProxyDescent):
            line_search_options["predicted_reduction"] = ft.partial(
                self.descent.predicted_reduction,
                descent_state=state.descent_state,
                args=args,
                options={},
            )

        line_sol = minimise(
            problem_1d,
            self.line_search,
            y0=state.next_init,
            args=args,
            options=line_search_options,
            max_steps=1000,
            throw=False,
            # max_steps=10,
            # throw=False,
        )
        (f_val, diff, new_aux, _, next_init) = line_sol.aux
        new_y = (ω(y) + ω(diff)).ω
        new_grad, _ = jax.jacrev(problem.fn, has_aux=problem.has_aux)(new_y, args)
        grad_diff = (ω(new_grad) - ω(state.vector)).ω
        hess_mv = state.operator.mv(diff)
        diff_outer = _outer(grad_diff)
        diff_inner = two_norm(grad_diff) ** 2
        hess_outer = _outer(hess_mv)
        hess_inner = tree_inner_prod(diff, hess_mv)
        term1 = (ω(diff_outer) / diff_inner).ω
        term2 = (ω(hess_outer) / hess_inner).ω
        new_hess = PyTreeLinearOperator(
            (state.operator.pytree**ω + term1**ω - term2**ω).ω,
            state.operator.out_structure(),
        )
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(diff) / ω(scale)).ω)
        descent_state = self.descent.update_state(
            state.descent_state, diff, new_grad, new_hess, {}
        )
        result = jnp.where(
            line_sol.result == RESULTS.max_steps_reached,
            RESULTS.successful,
            line_sol.result,
        )
        new_state = BFGSState(
            descent_state=descent_state,
            vector=new_grad,
            operator=new_hess,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=result,
            f_val=f_val,
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
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
        state: BFGSState,
    ):
        at_least_two = state.step >= 2
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

    def buffers(self, state: BFGSState):
        return ()
