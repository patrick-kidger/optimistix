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

import abc
from typing import Any, Callable, Generic, Optional, TypeVar

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree, Scalar

from ._custom_types import Aux, Fn, LineSearchAux, Out, SolverState, Y
from ._minimise import AbstractMinimiser
from ._solution import RESULTS


_DescentState = TypeVar("_DescentState")


class OneDimensionalFunction(eqx.Module, Generic[Y, Aux]):
    fn: Fn[Y, Scalar, Aux]
    descent: Callable
    y: Y

    def __call__(
        self, delta: Float[Array, ""], args: PyTree
    ) -> tuple[Scalar, LineSearchAux]:
        diff, result = self.descent(delta)
        new_y = (self.y**ω + diff**ω).ω
        f_val, aux = self.fn(new_y, args)
        return f_val, (f_val, diff, aux, result, jnp.array(0.0))


class AbstractLineSearch(AbstractMinimiser[SolverState, Scalar, LineSearchAux]):
    @abc.abstractmethod
    def first_init(
        self,
        vector: PyTree[Array],
        operator: lx.AbstractLinearOperator,
        options: dict[str, Any],
    ) -> Scalar:
        ...


class AbstractDescent(eqx.Module, Generic[_DescentState]):
    @abc.abstractmethod
    def init_state(
        self,
        fn: Fn[Y, Out, Aux],
        y: PyTree[Array],
        vector: PyTree[Array],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        args: Any,
        options: dict[str, Any],
    ) -> _DescentState:
        ...

    @abc.abstractmethod
    def update_state(
        self,
        descent_state: _DescentState,
        diff_prev: Optional[PyTree[Array]],
        vector: Optional[PyTree[Array]],
        operator: Optional[lx.AbstractLinearOperator],
        operator_inv: Optional[lx.AbstractLinearOperator],
        options: dict[str, Any],
    ) -> _DescentState:
        ...

    @abc.abstractmethod
    def __call__(
        self,
        delta: Scalar,
        descent_state: _DescentState,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS]:
        ...

    @abc.abstractmethod
    def predicted_reduction(
        self,
        diff: PyTree[Array],
        descent_state: _DescentState,
        args: PyTree,
        options: dict[str, Any],
    ) -> Scalar:
        ...
