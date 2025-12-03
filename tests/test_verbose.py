from collections import deque
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
import pytest


@pytest.fixture
def setup_callback():
    memory = deque()

    def callback(*args: tuple[bool, str, Any]):
        jax.debug.callback(memory.append, args)

    return memory, callback


@pytest.fixture
def test_fn():
    @eqx.filter_jit
    def fn(y, args):
        out = y
        aux = None
        return out, aux

    y = jnp.array(0.0)
    args = None
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    aux_struct = None
    tags = frozenset()
    options = {}

    return fn, (y, args, f_struct, aux_struct, tags, options)


def test_verbose_callback_gauss_newton(setup_callback, test_fn):
    memory, callback = setup_callback

    class SolverWithCallback(optx.GaussNewton):
        verbose_callback = callback

    callback_solver = SolverWithCallback(
        rtol=1e-6,
        atol=1e-6,
        verbose=frozenset({"step", "loss", "y"}),
    )

    fn, (y, args, f_struct, aux_struct, tags, options) = test_fn

    state = callback_solver.init(fn, y, args, options, f_struct, aux_struct, tags)

    callback_solver.step(fn, y, args, options, state, tags)

    assert len(memory) == 1

    entry = memory.pop()

    assert (jnp.array(True), "Step", jnp.array(0)) in entry
    assert (jnp.array(True), "Loss on this step", jnp.array(0.0)) in entry
    assert (jnp.array(True), "y", jnp.array(0.0)) in entry


def test_verbose_callback_lbfgs(setup_callback, test_fn):
    memory, callback = setup_callback

    class SolverWithCallback(optx.LBFGS):
        verbose_callback = callback

    callback_solver = SolverWithCallback(
        rtol=1e-6,
        atol=1e-6,
        verbose=frozenset({"loss", "y"}),
    )

    fn, (y, args, f_struct, aux_struct, tags, options) = test_fn

    state = callback_solver.init(fn, y, args, options, f_struct, aux_struct, tags)

    callback_solver.step(fn, y, args, options, state, tags)

    assert len(memory) == 1

    entry = memory.pop()

    assert (jnp.array(True), "Loss on this step", jnp.array(0.0)) in entry
    assert (jnp.array(True), "y", jnp.array(0.0)) in entry


def test_verbose_callback_optax(setup_callback, test_fn):
    memory, callback = setup_callback

    class SolverWithCallback(optx.OptaxMinimiser):
        verbose_callback = callback

    callback_solver = SolverWithCallback(
        rtol=1e-6,
        atol=1e-6,
        verbose=frozenset({"loss", "y"}),
        optim=optax.sgd(learning_rate=3e-3),
    )

    fn, (y, args, f_struct, aux_struct, tags, options) = test_fn

    state = callback_solver.init(fn, y, args, options, f_struct, aux_struct, tags)

    callback_solver.step(fn, y, args, options, state, tags)

    assert len(memory) == 1

    entry = memory.pop()
    assert (jnp.array(False), "Step", jnp.array(0)) in entry
    assert (jnp.array(True), "Loss", jnp.array(0.0)) in entry
    assert (jnp.array(True), "y", jnp.array(0.0)) in entry
