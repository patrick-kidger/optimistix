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
from typing import Any, TypeVar, Union
from typing_extensions import TypeAlias

import equinox.internal as eqxi


Args: TypeAlias = Any
Aux = TypeVar("Aux")
Out = TypeVar("Out")
SolverState = TypeVar("SolverState")
SearchState = TypeVar("SearchState")
DescentState = TypeVar("DescentState")
Y = TypeVar("Y")

Fn: TypeAlias = Callable[[Y, Args], tuple[Out, Aux]]
NoAuxFn: TypeAlias = Callable[[Y, Args], Out]
MaybeAuxFn: TypeAlias = Union[Fn[Y, Out, Aux], NoAuxFn[Y, Out]]

sentinel: Any = eqxi.doc_repr(object(), "sentinel")
