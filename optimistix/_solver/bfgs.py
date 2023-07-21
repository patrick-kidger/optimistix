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
from typing import Any, Generic, Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import AbstractLineSearchState, Aux, Fn, Y
from .._descent import AbstractLineSearch
from .._minimise import AbstractMinimiser, minimise
from .._misc import (
    AbstractHasTol,
    jacobian,
    max_norm,
    tree_dot,
    tree_full_like,
)
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent
from .misc import cauchy_termination


def _identity_pytree(pytree: PyTree[Array]) -> lx.PyTreeLinearOperator:
    """Create an identity pytree `I` such that
    `pytree = lx.PyTreeLinearOperator(I).mv(pytree)`

    **Arguments**:

    - `pytree`: A pytree such that the output of `_identity_pytree` is the identity
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

    # NOTE: this has a Lineax positive_semidefinite tag.
    # This is okay because the BFGS update preserves positive-definiteness.
    return lx.PyTreeLinearOperator(
        jtu.tree_unflatten(eye_structure, eye_leaves),
        jax.eval_shape(lambda: pytree),
        lx.positive_semidefinite_tag,
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
    operator: Optional[lx.PyTreeLinearOperator]
    operator_inv: Optional[lx.PyTreeLinearOperator]
    diff: Y
    f_val: Scalar
    f_prev: Scalar
    result: RESULTS


class BFGS(AbstractMinimiser[Y, Aux, _BFGSState[Y, Aux]], AbstractHasTol):
    """BFGS (Broyden--Fletcher--Goldfarb--Shanno) minimisation algorithm.

    This is a "second-order" optimisation algorithm, whose defining feature is that the
    second-order quantites are actually approximated using only first-order gradient
    information; this approximation is updated and improved at every step.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    use_inverse: bool = True
    # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
    line_search: AbstractLineSearch[
        Y, Aux, AbstractLineSearchState
    ] = BacktrackingArmijo(
        NewtonDescent(linear_solver=lx.Cholesky()),
        gauss_newton=False,
        backtrack_slope=0.1,
        decrease_factor=0.5,
    )

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
        if self.use_inverse:
            operator = None
            operator_inv = _identity_pytree(y)
        else:
            operator = _identity_pytree(y)
            operator_inv = None
        maxval = jnp.finfo(f_struct.dtype).max
        return _BFGSState(
            step_size=jnp.array(1.0),
            vector=tree_full_like(y, 1.0),
            operator=operator,
            operator_inv=operator_inv,
            diff=tree_full_like(y, jnp.inf),
            result=RESULTS.successful,
            f_val=jnp.array(maxval, f_struct.dtype),
            f_prev=jnp.array(0.5 * maxval, f_struct.dtype),
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
        inner = tree_dot(grad_diff, state.diff)
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
                operator_inner = tree_dot(grad_diff, inv_mvp)
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
                term2 = (_outer(mvp, mvp) ** ω / tree_dot(state.diff, mvp)).ω
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
            line_sol.result == RESULTS.nonlinear_max_steps_reached,
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


BFGS.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: The BFGS algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves). The
    default is (b), denoted via `use_inverse=True`. Note that this is incompatible with
    line search methods like [`optimistix.ClassicalTrustRegion`][], which use the
    Hessian approximation `B` as part of their own computations.
- `line_search`: The line search for the update. This can only require `options` from
    the list of:

        - "init_step_size"
        - "vector"
        - "operator"
        - "operator_inv"
        - "f0"
        - "fn"
        - "y"
"""
