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
from collections.abc import Callable
from typing import Any, Generic, Optional, TYPE_CHECKING

import equinox as eqx
import jax
from jaxtyping import Array, PyTree, Scalar

from ._adjoint import AbstractAdjoint
from ._custom_types import Aux, Fn, Out, SolverState, Y
from ._solution import Solution


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar


class AbstractSolver(eqx.Module, Generic[Y, Out, Aux, SolverState]):
    """Base class for all Optimistix solvers."""

    @abc.abstractmethod
    def solve(
        self,
        fn: Fn[Y, Out, Aux],
        y0: PyTree[Array],
        args: PyTree,
        options: dict[str, Any],
        *,
        max_steps: Optional[int],
        adjoint: AbstractAdjoint,
        throw: bool,
        tags: frozenset[object],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> Solution[Y, Aux, SolverState]:
        ...


class AbstractHasTol(AbstractSolver):
    """A solver guaranteed to have `atol` and `rtol` fields."""

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
