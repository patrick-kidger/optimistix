import functools as ft
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar

from ..custom_types import sentinel
from ..line_search import AbstractDescent, AbstractProxyDescent, OneDimensionalFunction
from ..linear_operator import AbstractLinearOperator, PyTreeLinearOperator
from ..minimise import AbstractMinimiser, minimise, MinimiseProblem
from ..misc import max_norm, tree_inner_prod
from ..solution import RESULTS
from .descent import UnnormalisedNewton, UnnormalisedNewtonInverse


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


def _std_basis(pytree):
    leaves, structure = jtu.tree_flatten(pytree)
    eye_structure = structure.compose(structure)
    eye_leaves = []
    for i1, l1 in enumerate(leaves):
        for i2, l2 in enumerate(leaves):
            if i1 == i2:
                eye_leaves.append(
                    jnp.eye(jnp.size(l1)).reshape(jnp.shape(l1) + jnp.shape(l2))
                )
            else:
                eye_leaves.append(jnp.zeros(l1.shape + l2.shape))
    return jtu.tree_unflatten(eye_structure, eye_leaves)


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
        descent: AbstractDescent = sentinel,
        norm: Callable = max_norm,
        converged_tol: float = 1e-2,
        use_inverse: bool = True,
    ):

        # # WARNING: if the user passes in a solver not compatible with use_inverse,
        # # then they will get very strange results and not a warning!
        if descent == sentinel:
            if use_inverse:
                descent = UnnormalisedNewtonInverse()
            else:
                descent = UnnormalisedNewton(gauss_newton=False)

        self.atol = atol
        self.rtol = rtol
        self.line_search = line_search
        self.descent = descent
        self.norm = norm
        self.converged_tol = converged_tol
        self.use_inverse = use_inverse
        self.descent = descent

    def init(
        self,
        problem: MinimiseProblem,
        y: PyTree[Array],
        args: Any,
        options: dict[str, Any],
    ):
        f0, aux = jtu.tree_map(
            lambda x: jnp.full(x.shape, jnp.inf, x.dtype),
            jax.eval_shape(problem.fn, y, args),
        )
        jrev = jax.jacrev(problem.fn, has_aux=True)
        vector, aux = jrev(y, args)
        # create an identity operator which we can update with BFGS
        # update/Woodbury identity
        operator = PyTreeLinearOperator(
            _std_basis(vector), output_structure=jax.eval_shape(lambda: vector)
        )
        descent_state = self.descent.init_state(problem, y, vector, operator, args, {})
        return BFGSState(
            descent_state=descent_state,
            vector=vector,
            operator=operator,
            diff=jtu.tree_map(lambda x: jnp.full(x.shape, jnp.inf, dtype=x.dtype), y),
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
        line_search_options = {
            "f0": state.f_val,
            "compute_f0": (state.step == 0),
            "vector": state.vector,
            "operator": state.operator,
            "diff": state.diff,
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
        )
        (f_val, diff, new_aux, _, next_init) = line_sol.aux
        new_y = (ω(y) + ω(diff)).ω
        new_grad, _ = jax.jacrev(problem.fn)(new_y, args)
        grad_diff = (ω(new_grad) - ω(state.vector)).ω
        inner = tree_inner_prod(grad_diff, diff)
        if self.use_inverse:
            # some complicated looking stuff, but it's just the application of
            # Woodbury identity to rank-1 update of approximate Hessian.
            operator_ip = tree_inner_prod(grad_diff, state.operator.mv(grad_diff))
            diff_outer = _outer(diff, diff)
            outer1 = _outer(state.operator.mv(grad_diff), diff)
            outer2 = _outer(diff, state.operator.transpose().mv(grad_diff))
            term1 = (((inner + operator_ip) * (diff_outer**ω)) / (inner**2)).ω
            term2 = ((outer1**ω + outer2**ω) / inner).ω
        else:
            diff_outer = _outer(grad_diff, grad_diff)
            hess_mv = state.operator.mv(diff)
            hess_outer = _outer(hess_mv, hess_mv)
            operator_ip = tree_inner_prod(diff, state.operator.mv(diff))
            term1 = (diff_outer**ω / inner).ω
            term2 = (hess_outer**ω / operator_ip).ω
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
        # WARNING: this is terminating 1 step too early on VariablyDimensioned
        # problem
        at_least_two = state.step >= 4
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
