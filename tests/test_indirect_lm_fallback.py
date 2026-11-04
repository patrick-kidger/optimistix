import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import pytest


@pytest.mark.parametrize("scale", [1.0, 100.0])
@pytest.mark.parametrize("lambda_0", [1.0, 1e8])
@pytest.mark.parametrize("root_finder", [optx.Newton, optx.Chord])
def test_indirect_lm_inner_root_fallback(scale, lambda_0, root_finder):
    solver = optx.IndirectLevenbergMarquardt(
        rtol=1e-8,
        atol=1e-8,
        lambda_0=lambda_0,
        root_finder=root_finder(rtol=1e-8, atol=1e-8),
    )
    sol = optx.least_squares(
        lambda y, _: scale * jnp.expm1(y),
        solver,
        jnp.array([-2.0]),
        max_steps=100,
        throw=False,
    )
    assert sol.result == optx.RESULTS.successful
    np.testing.assert_allclose(sol.value, 0, atol=1e-7)


@pytest.mark.parametrize("lambda_0", [1.0, 1e8])
def test_indirect_lm_complex_rotation(lambda_0):
    solver = optx.IndirectLevenbergMarquardt(
        rtol=1e-8,
        atol=1e-8,
        lambda_0=lambda_0,
        root_finder=optx.Chord(rtol=1e-8, atol=1e-8),
    )
    with pytest.warns(UserWarning, match="Complex support"):
        sol = optx.least_squares(
            lambda y, _: 1j * jnp.expm1(y / 1j),
            solver,
            jnp.array([-2j]),
            max_steps=150,
            throw=False,
        )
    assert sol.result == optx.RESULTS.successful
    np.testing.assert_allclose(sol.value, 0, atol=1e-7)


@pytest.mark.parametrize("norm", [optx.two_norm, optx.max_norm, optx.rms_norm])
def test_indirect_lm_indefinite_fallback(norm):
    # This is the local model of a bounded quartic objective at the origin.
    y = jnp.zeros(2)
    gradient = jnp.ones(2)
    hessian = lx.MatrixLinearOperator(
        jnp.diag(jnp.array([-1.0, 2.0])), tags=lx.symmetric_tag
    )
    info = optx.FunctionInfo.EvalGradHessian(jnp.array(0.0), gradient, hessian)
    descent = optx.IndirectDampedNewtonDescent(
        lambda_0=1e8,
        root_finder=optx.Chord(rtol=1e-8, atol=1e-8),
        trust_region_norm=norm,
    )
    state = descent.query(y, info, descent.init(y, eqx.filter_eval_shape(lambda: info)))
    step, result = descent.step(jnp.array(0.25), state)
    assert result == optx.RESULTS.successful
    assert jnp.dot(gradient, step) < 0
    np.testing.assert_allclose(norm(step), 0.25 * state.newton_norm, rtol=1e-10)
