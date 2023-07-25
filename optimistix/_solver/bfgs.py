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

from .._base_solver import AbstractHasTol
from .._custom_types import Aux, Fn, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    max_norm,
    NoAux,
    tree_dot,
    tree_full_like,
)
from .._search import AbstractDescent, AbstractSearch, DerivativeInfo
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent


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
                eye_leaves.append(jnp.zeros(jnp.shape(l1) + jnp.shape(l2)))

    # This has a Lineax positive_semidefinite tag. This is okay because the BFGS update
    # preserves positive-definiteness.
    return lx.PyTreeLinearOperator(
        jtu.tree_unflatten(eye_structure, eye_leaves),
        jax.eval_shape(lambda: pytree),
        lx.positive_semidefinite_tag,
    )


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


class _BFGSState(eqx.Module, Generic[Y, SearchState]):
    search_state: SearchState
    grad: Y
    hessian: Optional[lx.PyTreeLinearOperator]
    hessian_inv: Optional[lx.PyTreeLinearOperator]
    y_diff: Y
    f_val: Scalar
    f_prev: Scalar
    accept: Bool[Array, ""]
    result: RESULTS


class BFGS(AbstractMinimiser[Y, Aux, _BFGSState], AbstractHasTol):
    """BFGS (Broyden--Fletcher--Goldfarb--Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    use_inverse: bool = True
    descent: AbstractDescent[Y, Scalar, Any] = NewtonDescent(
        linear_solver=lx.Cholesky()
    )
    # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
    search: AbstractSearch[Y, Scalar, Any] = BacktrackingArmijo(
        slope=0.1, decrease_factor=0.5
    )

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BFGSState:
        del aux_struct
        init_search_state = self.search.init(self.descent, NoAux(fn), y, args, f_struct)
        if self.use_inverse:
            hessian = None
            hessian_inv = _identity_pytree(y)
        else:
            hessian = _identity_pytree(y)
            hessian_inv = None
        maxval = jnp.finfo(f_struct.dtype).max
        return _BFGSState(
            search_state=init_search_state,
            grad=tree_full_like(y, 0),
            hessian=hessian,
            hessian_inv=hessian_inv,
            f_val=jnp.array(maxval, f_struct.dtype),
            f_prev=jnp.array(0.5 * maxval, f_struct.dtype),
            y_diff=tree_full_like(y, 0),
            accept=jnp.array(True),
            result=RESULTS.successful,
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
        (f_val, aux), grad = jax.value_and_grad(fn, has_aux=True)(y, args)
        grad_diff = (grad**ω - state.grad**ω).ω
        inner = tree_dot(grad_diff, state.y_diff)

        dynamic, static = eqx.partition(
            (state.hessian, state.hessian_inv), eqx.is_array
        )

        def bfgs_update(dynamic):
            hessian, hessian_inv = eqx.combine(dynamic, static)
            if self.use_inverse:
                # Use Woodbury identity for rank-1 update of approximate Hessian.
                inv_mvp = hessian_inv.mv(grad_diff)
                mvp_inner = tree_dot(grad_diff, inv_mvp)
                diff_outer = _outer(state.y_diff, state.y_diff)
                mvp_outer = _outer(state.y_diff, inv_mvp)
                term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
                term2 = (
                    (_outer(inv_mvp, state.y_diff) ** ω + mvp_outer**ω) / inner
                ).ω

                hessian = None
                hessian_inv = lx.PyTreeLinearOperator(
                    (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,
                    output_structure=jax.eval_shape(lambda: y),
                    tags=lx.positive_semidefinite_tag,
                )
            else:
                # BFGS update to the operator directly, not inverse
                mvp = hessian.mv(state.y_diff)
                term1 = (_outer(grad_diff, grad_diff) ** ω / inner).ω
                term2 = (_outer(mvp, mvp) ** ω / tree_dot(state.y_diff, mvp)).ω
                hessian = lx.PyTreeLinearOperator(
                    (hessian.pytree**ω + term1**ω - term2**ω).ω,
                    output_structure=jax.eval_shape(lambda: y),
                    tags=lx.positive_semidefinite_tag,
                )
                hessian_inv = None

            return eqx.filter(hessian, eqx.is_array), eqx.filter(
                hessian_inv, eqx.is_array
            )

        def no_update(dynamic):
            return dynamic

        # In particular inner = 0 on the first step (as then state.grad=0), and so for
        # this we jump straight to the line search.
        # Likewise we get inner <= eps on convergence, and so again we make no update
        # to avoid a division by zero.
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps
        new_dynamic = lax.cond(inner_nonzero, bfgs_update, no_update, dynamic)
        hessian, hessian_inv = eqx.combine(new_dynamic, static)
        if self.use_inverse:
            deriv_info = DerivativeInfo.GradHessianInv(grad, hessian_inv)
        else:
            deriv_info = DerivativeInfo.GradHessian(grad, hessian)
        y_diff, accept, result, new_search_state = self.search.search(
            self.descent,
            NoAux(fn),
            y,
            args,
            f_val,
            state.search_state,
            deriv_info,
            max_steps=256,  # TODO(kidger): offer an API for this?
        )
        new_y = (y**ω + y_diff**ω).ω
        new_state = _BFGSState(
            search_state=new_search_state,
            grad=grad,
            hessian=hessian,
            hessian_inv=hessian_inv,
            y_diff=y_diff,
            f_val=f_val,
            f_prev=state.f_val,
            accept=accept,
            result=result,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BFGSState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            state.y_diff,
            state.f_val,
            state.f_prev,
            state.accept,
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
        - "aux"
"""
