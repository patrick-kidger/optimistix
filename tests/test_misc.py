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
import jax.flatten_util as jfu
import jax.numpy as jnp
import lineax as lx
import pytest

import optimistix as optx
import optimistix._misc as optx_misc

from .helpers import tree_allclose


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


def test_norm():
    def _square(x):
        return x * jnp.conj(x)

    def _two_norm(x):
        return jnp.sqrt(jnp.sum(_square(jfu.ravel_pytree(x)[0]))).real

    def _rms_norm(x):
        return jnp.sqrt(jnp.mean(_square(jfu.ravel_pytree(x)[0]))).real

    def _max_norm(x):
        return jnp.max(jnp.abs(jfu.ravel_pytree(x)[0]))

    x = [jnp.array(1.0), jnp.arange(4.0).reshape(2, 2)]
    tx = [jnp.array(0.5), jnp.arange(1.0, 5.0).reshape(2, 2)]

    assert jnp.allclose(optx.two_norm(x), _two_norm(x))
    assert jnp.allclose(optx.rms_norm(x), _rms_norm(x))
    assert jnp.allclose(optx.max_norm(x), _max_norm(x))

    two = jax.jvp(optx.two_norm, (x,), (tx,))
    true_two = jax.jvp(_two_norm, (x,), (tx,))
    rms = jax.jvp(optx.rms_norm, (x,), (tx,))
    true_rms = jax.jvp(_rms_norm, (x,), (tx,))
    max = jax.jvp(optx.max_norm, (x,), (tx,))
    true_max = jax.jvp(_max_norm, (x,), (tx,))
    assert tree_allclose(two, true_two)
    assert tree_allclose(rms, true_rms)
    assert tree_allclose(max, true_max)

    zero = [jnp.array(0.0), jnp.zeros((2, 2))]
    two0 = jax.jvp(optx.two_norm, (zero,), (tx,))
    rms0 = jax.jvp(optx.rms_norm, (zero,), (tx,))
    max0 = jax.jvp(optx.max_norm, (zero,), (tx,))
    true0 = (jnp.array(0.0), jnp.array(0.0))
    assert tree_allclose(two0, true0)
    assert tree_allclose(rms0, true0)
    assert tree_allclose(max0[0], true0[0])  # tangent rightly has nonzero value

    x = jnp.array([3 + 1.2j, -0.5 + 4.9j])
    tx = jnp.array([2 - 0.3j, -0.7j])
    two = jax.jvp(optx.two_norm, (x,), (tx,))
    true_two = jax.jvp(_two_norm, (x,), (tx,))
    rms = jax.jvp(optx.rms_norm, (x,), (tx,))
    true_rms = jax.jvp(_rms_norm, (x,), (tx,))
    max = jax.jvp(optx.max_norm, (x,), (tx,))
    true_max = jax.jvp(_max_norm, (x,), (tx,))
    assert two[0].imag == 0
    assert tree_allclose(two, true_two)
    assert rms[0].imag == 0
    assert tree_allclose(rms, true_rms)
    assert max[0].imag == 0
    assert tree_allclose(max, true_max)
