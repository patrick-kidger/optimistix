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

import dataclasses
import random

import jax
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray


jax.config.update("jax_enable_x64", True)


# This offers reproducability -- the initial seed is printed in the repr so we can see
# it when a test fails.
# Note the `eq=False`, which means that `_GetKey `objects have `__eq__` and `__hash__`
# based on object identity.
@dataclasses.dataclass(eq=False)
class _GetKey:
    seed: int
    call: int
    key: PRNGKeyArray

    def __init__(self, seed: int):
        self.seed = seed
        self.call = 0
        self.key = jr.PRNGKey(seed)

    def __call__(self):
        self.call += 1
        return jr.fold_in(self.key, self.call)


@pytest.fixture
def getkey():
    return _GetKey(random.randint(0, 2**31 - 1))
