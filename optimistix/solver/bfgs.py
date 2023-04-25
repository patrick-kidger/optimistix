from typing import Callable

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from ..line_search import AbstractDescent, AbstractLineSearch, OneDimProblem
from ..linear_operator import PyTreeLinearOperator
from ..minimise import minimise, MinimiseProblem
from ..misc import max_norm
from ..solution import RESULTS
from .descent import UnnormalizedNewton
from .misc import compute_hess_grad
from .quasi_newton import AbstractQuasiNewton, QNState


#
# Right now, this is performing the update `B_k -> B_(k + 1)` which solves
# `B_k p = g`. We could just as well update `Binv_k -> Binv_(k + 1)`, ie.
# `p = Binv_k g` is just the matrix vector product.
#


class BFGS(AbstractQuasiNewton):
    atol: float
    rtol: float
    norm: Callable

    def __init__(
        self,
        atol: float,
        rtol: float,
        line_search: AbstractLineSearch,
        descent: AbstractDescent = UnnormalizedNewton(),
        norm: Callable = max_norm,
    ):
        self.atol = atol
        self.rtol = rtol
        self.line_search = line_search
        self.descent = descent
        self.norm = norm

    def init(self, problem, y, args, options):

        vector, operator, aux = compute_hess_grad(problem, y, options, args)

        return QNState(
            step=jnp.array(0),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=jnp.array(RESULTS.successful),
            vector=vector,
            operator=operator,
            aux=aux,
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
            OneDimProblem(problem.fn, descent, y), has_aux=True
        )

        line_sol = minimise(
            problem_1d, self.line_search, jnp.array(2.0), args=args, options=options
        )

        (diff, new_aux) = line_sol.aux
        new_grad, _ = jax.jacrev(problem.fn, has_aux=problem.has_aux)(
            (ω(y) + ω(diff)).ω, args
        )

        t_diff = (ω(new_grad) - ω(state.vector)).ω

        def _outer(tree):
            def leaf_fn(x):
                return jtu.tree_map(lambda leaf2: jnp.tensordot(x, leaf2, axes=0), tree)

            return jtu.tree_map(leaf_fn, tree)

        hess_mv = state.operator.mv(diff)

        diff_outer = _outer(t_diff)
        diff_inner = jtu.tree_reduce(
            lambda x, y: x + y, (ω(t_diff) * ω(t_diff)).call(jnp.sum).ω
        )

        hess_outer = _outer(hess_mv)
        hess_inner = jtu.tree_reduce(
            lambda x, y: x + y, (ω(diff) * ω(hess_mv)).call(jnp.sum).ω
        )

        term1 = ω(diff_outer) / diff_inner
        term2 = ω(hess_outer) / hess_inner
        new_hess = PyTreeLinearOperator(
            (ω(state.operator.pytree) + term1 - term2).ω,
            jax.eval_shape(lambda: new_grad),
        )

        new_y = (ω(y) + ω(diff)).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(diff) / ω(scale)).ω)
        new_state = QNState(
            step=state.step + 1,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=line_sol.result,
            vector=new_grad,
            operator=new_hess,
            aux=new_aux,
        )

        return new_y, new_state, aux

    def buffer(self, state):
        return ()
