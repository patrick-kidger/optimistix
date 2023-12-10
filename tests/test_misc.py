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

import jax
import jax.numpy as jnp
import optimistix._misc as optx_misc


def test_inexact_asarray_no_copy():
    x = jnp.array([1.0])
    assert optx_misc.inexact_asarray(x) is x
    y = jnp.array([1.0, 2.0])
    assert jax.vmap(optx_misc.inexact_asarray)(y) is y


# See JAX issue #15676
def test_inexact_asarray_jvp():
    p, t = jax.jvp(optx_misc.inexact_asarray, (1.0,), (2.0,))
    assert type(p) is not float
    assert type(t) is not float
