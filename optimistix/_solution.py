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

from typing import Any, Generic

import equinox as eqx
import lineax as lx
from jaxtyping import ArrayLike, PyTree

from ._custom_types import Aux, Y


# Extend `lineax.RESULTS` as we want to be able to use their error messages too.
class RESULTS(lx.RESULTS):  # pyright: ignore
    successful = "Success."
    nonlinear_max_steps_reached = (
        "The maximum number of steps was reached in the nonlinear solver. "
        "The problem may not be solveable (e.g., a root-find on a function that has no "
        "roots), or you may need to increase `max_steps`."
    )
    nonlinear_divergence = "Nonlinear solve diverged."


class Solution(eqx.Module, Generic[Y, Aux]):
    """The solution to a nonlinear solve.

    **Attributes:**

    - `value`: The solution to the solve.
    - `result`: An integer representing whether the solve was successful or not. This
        can be converted into a human-readable error message via
        `optimistix.RESULTS[result]`
    - `aux`: Any user-specified auxiliary data returned from the problem; defaults to
        `None` if there is no auxiliary data. Auxiliary outputs can be captured by
        setting a `has_aux=True` flag, e.g. `optx.root_find(fn, ..., has_aux=True)`.
    - `stats`: Statistics about the solve, e.g. the number of steps that were required.
    - `state`: The final internal state of the solver. The meaning of this is specific
        to each solver.
    """

    value: Y
    result: RESULTS
    aux: Aux
    stats: dict[str, PyTree[ArrayLike]]
    state: Any
