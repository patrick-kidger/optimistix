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
import functools as ft
from typing import Callable, FrozenSet, Optional

import equinox as eqx
import equinox.internal as eqxi
import lineax as lx
from jaxtyping import Array, PyTree

from ._ad import implicit_jvp


class AbstractAdjoint(eqx.Module):
    @abc.abstractmethod
    def apply(
        self,
        primal_fn: Callable,
        rewrite_fn: Callable,
        inputs: PyTree,
        tags: FrozenSet[object],
    ) -> PyTree[Array]:
        ...


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    checkpoints: Optional[int] = None

    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        del rewrite_fn, tags
        while_loop = ft.partial(
            eqxi.while_loop, kind="checkpointed", checkpoints=self.checkpoints
        )
        return primal_fn(inputs, while_loop)


def _primal_fn(inputs):
    primal_fn, inputs, while_loop = inputs
    return primal_fn(inputs, while_loop)


class ImplicitAdjoint(AbstractAdjoint):
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=False)

    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        _inputs = (primal_fn, inputs, ft.partial(eqxi.while_loop, kind="lax"))
        return implicit_jvp(_primal_fn, rewrite_fn, _inputs, tags, self.linear_solver)
