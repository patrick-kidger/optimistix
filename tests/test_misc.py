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
import lineax as lx
import pytest

import optimistix as optx
import optimistix._misc as optx_misc


descent_linear = (
    (optx.NewtonDescent(), True),
    (optx.SteepestDescent(), True),
    (optx.NonlinearCGDescent(method=lambda v, v_p, d_p: jnp.array(0.0)), True),
    (optx.DoglegDescent(gauss_newton=False), False),
    (optx.DirectIterativeDual(gauss_newton=False), False),
    (optx.IndirectIterativeDual(gauss_newton=False, lambda_0=1.0), False),
)


@pytest.mark.parametrize("descent, expected_linear", descent_linear)
def test_linear_descents(descent, expected_linear):
    args = ()
    options = {
        "operator": jnp.array(16.0)
        * lx.IdentityLinearOperator(jax.ShapeDtypeStruct((), jnp.float64)),
        "vector": jnp.array(17.0),
        "vector_prev": jnp.array(17.0),
        "diff_prev": jnp.array(17.0),
    }

    def descent_no_results(descent, args, options, step_size):
        diff, results = descent(step_size, args, options)
        return diff

    linear = optx_misc.is_linear(
        lambda s: descent(s, args, options)[0], jnp.array(1.0), output=jnp.array(17.0)
    )
    assert linear == expected_linear


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
