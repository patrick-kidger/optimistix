# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft
from typing import Any, Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._line_search import AbstractDescent, AbstractLineSearch, OneDimensionalFunction
from .._minimise import AbstractMinimiser, minimise
from .._misc import (
    max_norm,
    tree_full,
    tree_full_like,
    tree_inner_prod,
    tree_where,
    tree_zeros,
    tree_zeros_like,
)
from .._solution import RESULTS
from .descent import UnnormalisedNewton


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


class _BFGSState(eqx.Module):
    descent_state: PyTree
    vector: PyTree[Array]
    operator: lx.PyTreeLinearOperator
    diff: PyTree[Array]
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    f_val: PyTree[Array]
    f_prev: PyTree[Array]
    next_init: Array
    aux: Any
    zero_inner_product: Bool[Array, ""]
    step: Scalar


#
# Right now, this is performing the update `B_k -> B_(k + 1)` which solves
# `B_k p = g`. We could just as well update `Binv_k -> Binv_(k + 1)`, ie.
# `p = Binv_k g` is just the matrix vector product.
#
class BFGS(AbstractMinimiser[_BFGSState, Y, Aux]):
    rtol: float
    atol: float
    line_search: AbstractLineSearch
    descent: AbstractDescent = UnnormalisedNewton(gauss_newton=False)
    norm: Callable = max_norm
    converged_tol: float = 1e-3
    use_inverse: bool = False

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: Any,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BFGSState:
        f0 = tree_full(f_struct, jnp.inf)
        aux = tree_zeros(aux_struct)
        # can this be removed?
        jrev = jax.jacrev(fn, has_aux=True)
        vector, aux = jrev(y, args)
        # create an identity operator which we can update with BFGS
        # update/Woodbury identity
        operator = lx.PyTreeLinearOperator(
            _std_basis(vector), output_structure=jax.eval_shape(lambda: vector)
        )
        if self.use_inverse:
            descent_state = self.descent.init_state(
                fn, y, vector, None, operator, args, options
            )
        else:
            descent_state = self.descent.init_state(
                fn, y, vector, operator, None, args, options
            )

        return _BFGSState(
            descent_state=descent_state,
            vector=vector,
            operator=operator,
            diff=jtu.tree_map(lambda x: jnp.full(x.shape, jnp.inf, dtype=x.dtype), y),
            diffsize=jnp.array(0.0),
            diffsize_prev=jnp.array(0.0),
            result=RESULTS.successful,
            f_val=f0,
            f_prev=f0,
            next_init=jnp.array(1.0),
            aux=aux,
            zero_inner_product=jnp.array(False),
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: Any,
        options: dict[str, Any],
        state: _BFGSState,
        tags: frozenset[object],
    ) -> tuple[Y, _BFGSState, Aux]:
        descent = eqx.Partial(
            self.descent,
            descent_state=state.descent_state,
            args=args,
            options=options,
        )
        problem_1d = OneDimensionalFunction(fn, descent, y)
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
        # Note that `options` at the solver level is passed to line_search init!
        # we anticipate a user may want to pass `options['init_line_search']` from
        # outside the solver.
        init = jnp.where(
            state.step == 0,
            self.line_search.first_init(state.vector, state.operator, options),
            state.next_init,
        )
        line_sol = minimise(
            fn=problem_1d,
            has_aux=True,
            solver=self.line_search,
            y0=init,
            args=args,
            options=line_search_options,
            max_steps=100,
            throw=False,
        )
        (f_val, diff, new_aux, _, next_init) = line_sol.aux
        new_y = (ω(y) + ω(diff)).ω
        new_grad, _ = jax.jacrev(fn)(new_y, args)
        grad_diff = (ω(new_grad) - ω(state.vector)).ω
        inner = tree_inner_prod(grad_diff, diff)
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps
        safe_inner = jnp.where(inner_nonzero, inner, jnp.array(1.0))
        raise_divergence = jnp.invert(inner_nonzero)

        def nonzero_inner(diff):
            if self.use_inverse:
                # some complicated looking stuff, but it's just the application of
                # Woodbury identity to rank-1 update of approximate Hessian.
                operator_ip = tree_inner_prod(grad_diff, state.operator.mv(grad_diff))
                diff_outer = _outer(diff, diff)
                outer1 = _outer(state.operator.mv(grad_diff), diff)
                outer2 = _outer(diff, state.operator.transpose().mv(grad_diff))
                term1 = tree_where(
                    inner_nonzero,
                    (((inner + operator_ip) * (diff_outer**ω)) / (safe_inner**2)).ω,
                    tree_full_like(diff_outer, jnp.inf),
                )
                term2 = tree_where(
                    inner_nonzero,
                    ((outer1**ω + outer2**ω) / safe_inner).ω,
                    tree_full_like(outer1, jnp.inf),
                )
            else:
                diff_outer = _outer(grad_diff, grad_diff)
                hess_mv = state.operator.mv(diff)
                hess_outer = _outer(hess_mv, hess_mv)
                operator_ip = tree_inner_prod(diff, state.operator.mv(diff))
                term1 = (diff_outer**ω / inner).ω
                term2 = (hess_outer**ω / operator_ip).ω
            scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
            diffsize = self.norm((diff**ω / scale**ω).ω)
            return term1, term2, diffsize

        def zero_inner(diff):
            # not sure this is any better than just materialising
            # `state.operator`.
            term1 = term2 = tree_zeros_like(_outer(diff, diff))
            return term1, term2, jnp.array(0.0)

        term1, term2, diffsize = lax.cond(
            inner_nonzero, nonzero_inner, zero_inner, diff
        )
        new_hess = lx.PyTreeLinearOperator(
            (state.operator.pytree**ω + term1**ω - term2**ω).ω,
            state.operator.out_structure(),
        )

        if self.use_inverse:
            descent_state = self.descent.update_state(
                state.descent_state, diff, new_grad, None, new_hess, options
            )
        else:
            descent_state = self.descent.update_state(
                state.descent_state, diff, new_grad, new_hess, None, options
            )
        result = RESULTS.where(
            line_sol.result == RESULTS.max_steps_reached,
            RESULTS.successful,
            line_sol.result,
        )
        new_state = _BFGSState(
            descent_state=descent_state,
            vector=new_grad,
            operator=new_hess,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=result,
            f_val=f_val,
            f_prev=state.f_val,
            next_init=next_init,
            aux=new_aux,
            zero_inner_product=raise_divergence,
            step=state.step + 1,
        )

        # notice that this is state.aux, not new_state.aux or aux.
        # we assume aux is the aux at f(y), but line_search returns
        # aux at f(y_new), which is new_state.aux. So, we simply delay
        # the return of aux for one eval
        return new_y, new_state, state.aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: Any,
        options: dict[str, Any],
        state: _BFGSState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        at_least_two = state.step >= 2
        f_diff = jnp.abs(state.f_val - state.f_prev)
        converged = f_diff < self.rtol * jnp.abs(state.f_prev) + self.atol
        linsolve_fail = state.result != RESULTS.successful
        terminate = (
            linsolve_fail | state.zero_inner_product | (converged & at_least_two)
        )
        result = RESULTS.where(linsolve_fail, state.result, RESULTS.successful)
        return terminate, result

    def buffers(self, state: _BFGSState) -> tuple[()]:
        return ()
