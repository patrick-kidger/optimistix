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

from typing import Any, Optional

import equinox as eqx
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._line_search import AbstractDescent
from .._misc import tree_inner_prod, two_norm
from .._solution import RESULTS
from .misc import quadratic_predicted_reduction


class _GradientState(eqx.Module):
    vector: PyTree[Array]


class UnnormalisedGradient(AbstractDescent[_GradientState]):
    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: PyTree,
        options: dict[str, Any],
    ) -> _GradientState:
        return _GradientState(vector)

    def update_state(
        self,
        descent_state: _GradientState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ) -> _GradientState:
        return _GradientState(vector)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _GradientState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        return (-delta * descent_state.vector**ω).ω, RESULTS.successful

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _GradientState,
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        return tree_inner_prod(descent_state.vector, diff)


class NormalisedGradient(AbstractDescent[_GradientState]):
    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: PyTree,
        options: dict[str, Any],
    ) -> _GradientState:
        return _GradientState(vector)

    def update_state(
        self,
        descent_state: _GradientState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ) -> _GradientState:
        return _GradientState(vector)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _GradientState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        diff = ((-delta * descent_state.vector**ω) / two_norm(descent_state.vector)).ω
        return diff, RESULTS.successful

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _GradientState,
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        return tree_inner_prod(descent_state.vector, diff)


#
# NOTE: we handle both the Gauss-Newton and quasi-Newton case identically
# in `__call__`. This is because the quasi-Newton step is
# `-B^† g` where`B^†` is the Moore-Penrose pseudoinverse of the quasi-Newton
# matrix `B` and `g` the gradient of the function to minimise.
# In the Gauss-Newton setting the step is `-J^† r` where
# `J^†` is the pseudoinverse of the Jacobian and `r` is the residual vector.
# Throughout, we have abstracted `J` and `B` into `operator` and
# `g` and `r` into `vector`, so the solves are identical.
#
# However, note that the gauss_newton flag is still necessary for the
# predicted reduction, whos computation does change depending on
# whether `operator` and `vector` contains the quasi-Newton matrix and
# vector or the Jacobian and residual. In line-searches which do not use
# this, there is no difference between `gauss_newton=True` and `gauss_newton=False`.
#


class _NewtonState(eqx.Module):
    vector: PyTree[Array]
    operator: Optional[lx.AbstractLinearOperator]
    operator_inv: Optional[lx.AbstractLinearOperator]


class UnnormalisedNewton(AbstractDescent[_NewtonState]):
    gauss_newton: bool = False

    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: PyTree,
        options: dict[str, Any],
    ) -> _NewtonState:
        return _NewtonState(vector, operator, operator_inv)

    def update_state(
        self,
        descent_state: _NewtonState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ) -> _NewtonState:
        return _NewtonState(vector, operator, operator_inv)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _NewtonState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        if descent_state.operator_inv is not None:
            newton = descent_state.operator_inv.mv(descent_state.vector)
            result = RESULTS.successful
        elif descent_state.operator is not None:
            out = lx.linear_solve(
                descent_state.operator,
                descent_state.vector,
                lx.AutoLinearSolver(well_posed=False),
            )
            newton = out.value
            result = RESULTS.promote(out.result)
        else:
            raise ValueError(
                "At least one of `operator` or `operator_inv` must be "
                "passed to the UnnormalisedNewton descent."
            )
        return (-delta * newton**ω).ω, result

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _NewtonState,
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )


class NormalisedNewton(AbstractDescent[_NewtonState]):
    gauss_newton: bool = False

    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: PyTree,
        options: dict[str, Any],
    ) -> _NewtonState:
        return _NewtonState(vector, operator, operator_inv)

    def update_state(
        self,
        descent_state: _NewtonState,
        diff_prev: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ) -> _NewtonState:
        return _NewtonState(vector, operator, operator_inv)

    def __call__(
        self,
        delta: Scalar,
        descent_state: _NewtonState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        if descent_state.operator_inv is not None:
            newton = descent_state.operator_inv.mv(descent_state.vector)
            result = RESULTS.successful
        elif descent_state.operator is not None:
            out = lx.linear_solve(
                descent_state.operator,
                descent_state.vector,
                lx.AutoLinearSolver(well_posed=False),
            )
            newton = out.value
            result = RESULTS.promote(out.result)
        else:
            raise ValueError(
                "At least one of `operator` or `operator_inv` must be "
                "passed to the UnnormalisedNewton descent."
            )
        return ((-delta * newton**ω) / two_norm(newton)).ω, result

    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _NewtonState,
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        return quadratic_predicted_reduction(
            self.gauss_newton, diff, descent_state, args, options
        )
