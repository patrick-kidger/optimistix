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
from typing import Any, Generic, TYPE_CHECKING

import equinox as eqx
from jaxtyping import PyTree, Scalar


if TYPE_CHECKING:
    pass
else:
    pass

from ._custom_types import Aux, LineSearchState, Y
from ._minimise import AbstractMinimiser
from ._solution import RESULTS


class AbstractDescent(eqx.Module, Generic[Y]):
    """The abstract base class for descents. A descent is a method which consumes a
    scalar, and returns the `diff` to take at point `y`, so that `y + diff` is the next
    iterate in a nonlinear optimisation problem.

    This generalises the concept of line search and trust region to anything which
    takes a step-size and returns the step to take given that step-size.
    """

    @abc.abstractmethod
    def __call__(
        self,
        step_size: Scalar,
        args: PyTree,
        options: dict[str, Any],
    ) -> tuple[Y, RESULTS]:
        """Computes the descent direction.

        !!! warning

            This API is not stable and may change in the future.

            Right now the `options` dictionary is essentially unstructured, but figuring
            out a nicer type hierarachy to describe things seems pretty complicated. We
            may or may not do this in the future.

        **Arguments:**

        - `step_size`: a non-negative scalar describing the step size to take.
        - `args`: as passed to e.g. `optx.least_squares(..., args=...)`. Is just
            forwarded on to the user function.
        - `options`: an unstructured dictionary of "everything else" that the caller
            makes available for the descent, and which the descent can use to determine
            its output.

        **Returns:**

        The step to take in y-space.
        """


class AbstractLineSearch(AbstractMinimiser[LineSearchState, Y, Aux]):
    """The abstract base class for all line searches.

    In practice these are all just minimisers, and this class really just exists to help
    group together those minimisers with line-search-like behaviour. (That they don't
    try to find an actual minimum -- just something better than the initial point.)
    """
