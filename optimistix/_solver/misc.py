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

from typing import Any

import jax
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._custom_types import Aux, Fn, Out, Y
from .._misc import jacobian, sum_squares, tree_inner_prod


def compute_hess_grad(fn: Fn[Y, Out, Aux], y: PyTree[Array], args: PyTree):
    in_size = jtu.tree_reduce(lambda a, b: a + b, jtu.tree_map(lambda c: c.size, y))
    jac = jacobian(fn, in_size, out_size=1, has_aux=True)
    grad, aux = jac(y, args)
    hessian_mat, _ = jax.jacfwd(jac, has_aux=True)(y, args)
    hessian = lx.PyTreeLinearOperator(
        hessian_mat,
        output_structure=jax.eval_shape(lambda: grad),
    )
    return grad, hessian, aux


def compute_jac_residual(fn: Fn[Y, Out, Aux], y: PyTree[Array], args: PyTree):
    residual, aux = fn(y, args)
    jacobian = lx.JacobianLinearOperator(fn, y, args, _has_aux=True)
    return residual, jacobian, aux


def quadratic_predicted_reduction(
    gauss_newton: bool,
    diff: PyTree[Array],
    descent_state: PyTree,
    args: PyTree,
    options: dict[str, Any],
):
    # The predicted reduction of a quadratic model function.
    # This is the model quadratic model function of classical trust region
    # methods localized around f(x). ie. `m(p) = g^t p + 1/2 p^T B p`
    # where `g` is the gradient, `B` the Quasi-Newton approximation to the
    # Hessian, and `p` the descent direction (diff).
    #
    # In the Gauss-Newton setting we compute
    # `0.5 * [(Jp + r)^T (Jp + r) - r^T r]`
    # which is equivalent when `B = J^T J` and `g = J^T r`.
    if descent_state.operator is None:
        raise ValueError(
            "Cannot get predicted reduction without `operator`. "
            "`operator_inv` cannot be used with predicted_reduction."
        )
    if gauss_newton:
        rtr = sum_squares(descent_state.vector)
        jacobian_term = sum_squares(
            (ω(descent_state.operator.mv(diff)) + ω(descent_state.vector)).ω
        )
        reduction = 0.5 * (jacobian_term - rtr)
    else:
        operator_quadratic = 0.5 * tree_inner_prod(
            diff, descent_state.operator.mv(diff)
        )
        steepest_descent = tree_inner_prod(descent_state.vector, diff)
        reduction = (operator_quadratic**ω + steepest_descent**ω).ω
    return reduction
