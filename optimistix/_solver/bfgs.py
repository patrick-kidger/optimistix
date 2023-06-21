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

from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import AbstractLineSearchState, Aux, Fn, Y
from .._minimise import AbstractMinimiser, minimise
from .._misc import (
    jacobian,
    max_norm,
    tree_full_like,
    tree_inner_prod,
)
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent
from .misc import cauchy_termination


def _identity_pytree(pytree: PyTree[Array]) -> lx.PyTreeLinearOperator:
    """Create an identity pytree `I` such that
    `pytree = lx.PyTreeLinearOperator(I).mv(pytree)`

    **Arguments**:

        - `pytree`: a pytree such that the output of `_identity_pytree` is the identity
        with respect to pytrees of the same shape as `pytree`.

    **Returns**:

    A `lx.PyTreeLinearOperator` with input and output shape the shape of `pytree`.
    """
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
    return lx.PyTreeLinearOperator(
        jtu.tree_unflatten(eye_structure, eye_leaves), jax.eval_shape(lambda: pytree)
    )


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


def _auxmented(fn, x, args):
    f_val, aux = fn(x, args)
    return f_val, (f_val, aux)


class _BFGSState(eqx.Module, Generic[Y, Aux]):
    step_size: Scalar
    vector: Y
    operator: lx.PyTreeLinearOperator
    operator_inv: lx.PyTreeLinearOperator
    diff: Y
    f_val: Scalar
    f_prev: Scalar
    result: RESULTS


class AbstractBFGS(AbstractMinimiser[_BFGSState[Y, Aux], Y, Aux]):
    rtol: float
    atol: float
    line_search: AbstractMinimiser[AbstractLineSearchState, Y, Aux]
    norm: Callable
    use_inverse: bool

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BFGSState[Y, Aux]:
        del fn, aux_struct
        identity_pytree_operator = _identity_pytree(y)
        if self.use_inverse:
            operator = None
            operator_inv = identity_pytree_operator
        else:
            operator = identity_pytree_operator
            operator_inv = None
        return _BFGSState(
            step_size=jnp.array(1.0),
            vector=tree_full_like(y, jnp.array(1.0)),
            operator=operator,
            operator_inv=operator_inv,
            diff=tree_full_like(y, jnp.inf),
            result=RESULTS.successful,
            f_val=jnp.array(jnp.inf, f_struct.dtype),
            f_prev=jnp.array(jnp.inf, f_struct.dtype),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BFGSState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Y, _BFGSState[Y, Aux], Aux]:
        in_size = jtu.tree_reduce(lambda a, b: a + b, jtu.tree_map(lambda c: c.size, y))
        new_grad, (f_val, aux) = jacobian(
            eqx.Partial(_auxmented, fn), in_size, out_size=1, has_aux=True
        )(y, args)
        grad_diff = (ω(new_grad) - ω(state.vector)).ω
        # On the first iteration, `state.diff = jnp.inf` and
        # so anything divided by `inner` is just 0.
        _finite = lambda x: jnp.all(jnp.isfinite(x))
        diff_finite = jtu.tree_reduce(
            lambda x, y: x | y, jtu.tree_map(_finite, state.diff)
        )
        inner = tree_inner_prod(grad_diff, state.diff)
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps

        operator_dynamic, operator_static = eqx.partition(state.operator, eqx.is_array)
        operator_inv_dynamic, operator_inv_static = eqx.partition(
            state.operator_inv, eqx.is_array
        )

        def bfgs_update(operator_and_inv_dynamic):
            operator_dynamic, operator_inv_dynamic = operator_and_inv_dynamic
            operator = eqx.combine(operator_dynamic, operator_static)
            operator_inv = eqx.combine(operator_inv_dynamic, operator_inv_static)
            if self.use_inverse:
                # Use Woodbury identity for rank-1 update of approximate Hessian.
                inv_mvp = operator_inv.mv(grad_diff)
                operator_inner = tree_inner_prod(grad_diff, inv_mvp)
                diff_outer = _outer(state.diff, state.diff)
                mvp_outer = _outer(state.diff, operator_inv.transpose().mv(grad_diff))
                term1 = (
                    ((inner + operator_inner) * (diff_outer**ω)) / (inner**2)
                ).ω
                term2 = ((_outer(inv_mvp, state.diff) ** ω + mvp_outer**ω) / inner).ω

                operator = None
                operator_inv = lx.PyTreeLinearOperator(
                    (operator_inv.pytree**ω + term1**ω - term2**ω).ω,
                    output_structure=jax.eval_shape(lambda: y),
                )
            else:
                # BFGS update to the operator directly, not inverse
                mvp = operator.mv(state.diff)
                term1 = (_outer(grad_diff, grad_diff) ** ω / inner).ω
                term2 = (_outer(mvp, mvp) ** ω / tree_inner_prod(state.diff, mvp)).ω
                operator = lx.PyTreeLinearOperator(
                    (operator.pytree**ω + term1**ω - term2**ω).ω,
                    output_structure=jax.eval_shape(lambda: y),
                )
                operator_inv = None

            return eqx.filter(operator, eqx.is_array), eqx.filter(
                operator_inv, eqx.is_array
            )

        def no_update(operator_and_inv_dynamic):
            operator_dynamic, operator_inv_dynamic = operator_and_inv_dynamic
            return operator_dynamic, operator_inv_dynamic

        # Typically `inner = 0` implies that we have converged, so we do an identity
        # update and terminate.
        #
        # Pass and unpack the tuple `(operator_dynamic, operator_inv_dynamic)` rather
        # than directly pass `operator_dynamic` and `operator_inv_dynamic` to work
        # around JAX issue #16413
        new_operator_dynamic, new_operator_inv_dynamic = lax.cond(
            diff_finite & inner_nonzero,
            bfgs_update,
            no_update,
            (operator_dynamic, operator_inv_dynamic),
        )
        operator = eqx.combine(new_operator_dynamic, operator_static)
        operator_inv = eqx.combine(new_operator_inv_dynamic, operator_inv_static)
        line_search_options = {
            "init_step_size": state.step_size,
            "vector": new_grad,
            "operator": operator,
            "operator_inv": operator_inv,
            "f0": f_val,
            "fn": fn,
            "y": y,
        }
        line_sol = minimise(
            fn,
            self.line_search,
            y,
            args,
            line_search_options,
            has_aux=True,
            throw=False,
        )
        new_y = line_sol.value
        result = RESULTS.where(
            line_sol.result == RESULTS.max_steps_reached,
            RESULTS.successful,
            line_sol.result,
        )
        new_state = _BFGSState(
            step_size=line_sol.state.next_init,
            vector=new_grad,
            operator=operator,
            operator_inv=operator_inv,
            diff=(new_y**ω - y**ω).ω,
            f_val=f_val,
            f_prev=state.f_val,
            result=result,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BFGSState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            state.diff,
            state.f_val,
            state.f_prev,
            state.result,
        )

    def buffers(self, state: _BFGSState[Y, Aux]) -> tuple[()]:
        return ()


class BFGS(AbstractBFGS):
    def __init__(self, rtol: float, atol: float, use_inverse=True):
        self.rtol = rtol
        self.atol = atol
        # TODO(raderj): switch out `BacktrackingArmijo` with a better
        # line search.
        self.line_search = BacktrackingArmijo(
            NewtonDescent(),
            gauss_newton=False,
            backtrack_slope=0.1,
            decrease_factor=0.5,
        )
        self.norm = max_norm
        self.use_inverse = use_inverse
