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

import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Scalar

from .._custom_types import Y
from .._solution import RESULTS


def cauchy_termination(
    rtol: float,
    atol: float,
    norm: Callable,
    y: Y,
    diff: Y,
    f_val: Scalar,
    f_prev: Scalar,
    result: RESULTS,
) -> tuple[Bool[Array, ""], RESULTS]:
    """Terminate if there is a small difference in both `y` space and `f` space, as
    determined by `rtol` and `atol`.

    Specifically, this checks that `y_difference < atol + rtol * y` and
    `f_difference < atol + rtol * f_prev`, terminating if both of these are true.
    """
    y_scale = (atol + rtol * ω(y).call(jnp.abs)).ω
    y_converged = norm((ω(diff).call(jnp.abs) / y_scale**ω).ω) < 1
    f_scale = (atol + rtol * ω(f_prev).call(jnp.abs)).ω
    f_converged = norm(((f_val**ω - f_prev**ω).call(jnp.abs) / f_scale**ω).ω) < 1
    terminate = (result != RESULTS.successful) | (y_converged & f_converged)
    return terminate, result
