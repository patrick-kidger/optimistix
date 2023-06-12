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
    jacobian,
    max_norm,
    tree_full,
    tree_full_like,
    tree_inner_prod,
    tree_where,
    tree_zeros,
    tree_zeros_like,
)
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .descent import UnnormalisedNewton


def _std_basis(pytree: PyTree[Array]) -> PyTree[Array]:
    # Create an "identity pytree" `out` such that
    # `pytree = lx.PyTreeLinearOperator(out).mv(pytree)`
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


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


class _BFGSState(eqx.Module):
    descent_state: PyTree
    vector: PyTree[Array]
    operator: lx.PyTreeLinearOperator
    diff: PyTree[Array]
    result: RESULTS
    f_val: PyTree[Array]
    f_prev: PyTree[Array]
    next_init: Array
    aux: Any
    zero_inner_product: Bool[Array, ""]
    size: Scalar
    step: Scalar


# TODO(raderj): switch out `BacktrackingArmijo` with a better
# line search.
class BFGS(AbstractMinimiser[_BFGSState, Y, Aux]):
    rtol: float
    atol: float
    line_search: AbstractLineSearch = BacktrackingArmijo(
        gauss_newton=False, backtrack_slope=0.1, decrease_factor=0.5
    )
    descent: AbstractDescent = UnnormalisedNewton(gauss_newton=False)
    norm: Callable = max_norm
    converged_tol: float = 1e-3
    use_inverse: bool = False

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BFGSState:
        f0 = tree_full(f_struct, jnp.inf)
        aux = tree_zeros(aux_struct)
        # This size is for passing to Jacobian later. It's easier to store the
        # scalar than recompute it repeatedly.
        size = jtu.tree_reduce(lambda a, b: a + b, jtu.tree_map(lambda c: c.size, y))
        # Dummy vector and operator for first pass. Note that having `jnp.ones_like`
        # is preferable to `jnp.zeros_like` as the latter can lead to linear solves
        # of the form `0 x = 0` which can return `nan` values.
        vector = jtu.tree_map(jnp.ones_like, y)
        # `_std_basis` creates an identity operator which we can update with BFGS
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
            result=RESULTS.successful,
            f_val=f0,
            f_prev=f0,
            next_init=jnp.array(1.0),
            aux=aux,
            zero_inner_product=jnp.array(False),
            size=size,
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
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
        # For step 0 we are just getting the vector
        # and don't care about the line search. So we want to pass `compute_f0 = True`
        # on the first step where we actually care. We pass `f0=jnp.inf` the first two
        # steps because we don't have an accurate value of `f0` until we have the
        # correct `vector`/`operator`.
        line_search_options = {
            "f0": jnp.where(state.step > 1, state.f_val, jnp.inf),
            "compute_f0": (state.step == 1),
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
        # Note: this `<=1` is atypical, but since we are passing `vector = 0`
        # on the first iteration, there is a chance that the line search inits to
        # 0 and we get stuck there.
        init = jnp.where(
            state.step <= 1,
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
        # `new_aux` and `f_val` are the output of of at `f` at step the
        # end of the line search. ie. they are not `f(y)`, but rather
        # `f(y_new)` where `y_new` is `y` in the next call of `step`. In
        # other words, we use FSAL.
        (f_val, diff, new_aux, _, next_init) = line_sol.aux
        new_y = tree_where(state.step > 0, (ω(y) + ω(diff)).ω, y)
        fn_grad = jacobian(fn, in_size=state.size, out_size=1, has_aux=True)
        new_grad, _ = fn_grad(new_y, args)
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
                term1 = tree_where(
                    inner_nonzero,
                    (((inner + operator_ip) * (diff_outer**ω)) / (safe_inner**2)).ω,
                    tree_full_like(diff_outer, jnp.inf),
                )

                outer1 = _outer(state.operator.mv(grad_diff), diff)
                outer2 = _outer(diff, state.operator.transpose().mv(grad_diff))
                term2 = tree_where(
                    inner_nonzero,
                    ((outer1**ω + outer2**ω) / safe_inner).ω,
                    tree_full_like(outer1, jnp.inf),
                )
            else:
                # BFGS update to the operator directly, not inverse
                grad_diff_outer = _outer(grad_diff, grad_diff)
                hess_mv = state.operator.mv(diff)
                hess_outer = _outer(hess_mv, hess_mv)
                operator_ip = tree_inner_prod(diff, state.operator.mv(diff))
                term1 = (grad_diff_outer**ω / inner).ω
                term2 = (hess_outer**ω / operator_ip).ω
            return term1, term2

        def zero_inner(diff):
            _outer(diff, diff)
            term1 = term2 = tree_zeros_like(_outer(diff, diff))
            return term1, term2

        # This serves two purposes!
        # 1. In the init we set vector to a dummy value to save on a compilation of
        # `fn`. Because of this, we need to compute the actual `vector` before
        # taking any iterates, so we make no update to the BFGS matrix.
        # 2. If the `grad_diff.T @ diff = 0` just set `term1 = term2 = 0` as all
        term1, term2 = lax.cond(
            inner_nonzero & (state.step > 0), nonzero_inner, zero_inner, diff
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
            (line_sol.result == RESULTS.max_steps_reached),
            RESULTS.successful,
            line_sol.result,
        )
        new_state = _BFGSState(
            descent_state=descent_state,
            vector=new_grad,
            operator=new_hess,
            diff=diff,
            result=result,
            f_val=f_val,
            f_prev=state.f_val,
            next_init=next_init,
            aux=new_aux,
            zero_inner_product=raise_divergence,
            size=state.size,
            step=state.step + 1,
        )

        # Notice that this is `state.aux`, not `new_state.aux` or `aux`.
        # we delay the return of `aux` by one step because of the FSAL
        # in the line search.
        # We want aux at `f(y)`, but line_search returns
        # `aux` at `f(y_new)`
        return new_y, new_state, state.aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BFGSState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        at_least_two = state.step >= 2
        y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
        y_converged = self.norm((state.diff**ω / y_scale**ω).ω) < 1
        f_scale = self.rtol * jnp.abs(state.f_prev) + self.atol
        f_converged = (jnp.abs(state.f_val - state.f_prev) / f_scale) < 1
        converged = y_converged & f_converged
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (converged & at_least_two)
        result = RESULTS.where(linsolve_fail, state.result, RESULTS.successful)
        return terminate, result

    def buffers(self, state: _BFGSState) -> tuple[()]:
        return ()
