import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
import scipy.optimize


def _rosenbrock(y, args):
    return jnp.stack([10 * (y[1] - y[0] ** 2), 1 - y[0]])


def _decay(y, args):
    # `y[2]` enters a thousand times weaker than the others: a badly-scaled problem.
    times, data = args
    return y[0] * jnp.exp(-y[1] * times) + 1e-3 * y[2] - data


_times = jnp.linspace(0.0, 5.0, 40)
_data = 3.0 * jnp.exp(-0.7 * _times) + 0.2 + 0.01 * jnp.sin(9.0 * _times)
_problems = [
    (_rosenbrock, jnp.array([-1.2, 1.0]), None),
    (_decay, jnp.array([1.0, 3.0, 50.0]), (_times, _data)),
]


def _scipy(fn, y0, args, **kwargs):
    residual = lambda y: np.asarray(fn(jnp.asarray(y), args))
    jacobian = lambda y: np.asarray(jax.jacfwd(fn)(jnp.asarray(y), args))
    return scipy.optimize.least_squares(
        residual,
        np.asarray(y0),
        jac=jacobian,  # pyright: ignore[reportArgumentType]  (a stub types it as `str`)
        method="trf",
        **kwargs,
    )


@pytest.mark.parametrize("fn, y0, args", _problems)
@pytest.mark.parametrize("x_scale", ["jac", 1.0, 0.5])
def test_matches_scipy_evaluation_by_evaluation(fn, y0, args, x_scale):
    # One step is one evaluation of `fn`, as `max_nfev` counts them in SciPy, so the two
    # must agree after every budget -- not just at the end.
    tol = 1e-12
    for budget in range(1, 30):
        ours = optx.least_squares(
            fn,
            optx.TRF(tol, tol, x_scale=x_scale),
            y0,
            args,
            max_steps=budget,
            throw=False,
        )
        theirs = _scipy(
            fn, y0, args, x_scale=x_scale, max_nfev=budget, ftol=tol, xtol=tol, gtol=tol
        )
        np.testing.assert_allclose(ours.value, theirs.x, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("fn, y0, args", _problems)
def test_converges_in_as_many_evaluations_as_scipy(fn, y0, args):
    ours = optx.least_squares(fn, optx.TRF(1e-8, 1e-8), y0, args)
    theirs = _scipy(fn, y0, args, x_scale="jac")
    assert ours.result == optx.RESULTS.successful
    assert int(ours.stats["num_steps"]) == theirs.nfev
    np.testing.assert_allclose(ours.value, theirs.x, rtol=1e-10)


def test_a_nonfinite_trial_shrinks_the_region_and_is_retried():
    # The residual is undefined past y = 2, and the full Gauss--Newton step from the
    # start lands there.
    def fn(y, args):
        return jnp.stack([jnp.log(2.0 - y[0]) - jnp.log(1.9), 1e-2 * y[1]])

    y0 = jnp.array([-5.0, 1.0])
    ours = optx.least_squares(fn, optx.TRF(1e-10, 1e-10), y0, max_steps=100)
    theirs = _scipy(fn, y0, None, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10)
    assert ours.result == optx.RESULTS.successful
    np.testing.assert_allclose(ours.value, theirs.x, rtol=1e-8, atol=1e-12)
    assert int(ours.stats["num_steps"]) == theirs.nfev


def test_a_nonfinite_start_is_reported():
    fn = lambda y, args: jnp.log(y) - 1.0
    sol = optx.least_squares(fn, optx.TRF(1e-8, 1e-8), jnp.array([-1.0]), throw=False)
    assert sol.result == optx.RESULTS.nonfinite


def test_a_stationary_start_stops_on_the_gradient():
    fn = lambda y, args: jnp.stack([y[0] - 1.0, y[1] + 2.0])
    sol = optx.least_squares(fn, optx.TRF(1e-8, 1e-8), jnp.array([1.0, -2.0]))
    assert int(sol.stats["num_steps"]) == 1


def test_forward_and_reverse_jacobians_agree():
    fn, y0, args = _problems[1]
    solver = optx.TRF(1e-10, 1e-10)
    fwd = optx.least_squares(fn, solver, y0, args, options=dict(jac="fwd"))
    bwd = optx.least_squares(fn, solver, y0, args, options=dict(jac="bwd"))
    np.testing.assert_allclose(fwd.value, bwd.value, rtol=1e-10)


def test_x_scale_as_a_pytree():
    def fn(y, args):
        return jnp.stack([10 * (y["b"][0] - y["a"] ** 2), 1 - y["a"]])

    y0 = {"a": jnp.array(-1.2), "b": jnp.array([1.0])}
    x_scale = {"a": 2.0, "b": jnp.array([0.5])}
    ours = optx.least_squares(
        fn, optx.TRF(1e-12, 1e-12, x_scale=x_scale), y0, max_steps=8, throw=False
    )
    flat = lambda y: _rosenbrock(jnp.stack([y[0], y[1]]), None)
    theirs = _scipy(
        lambda y, args: flat(y),
        jnp.array([-1.2, 1.0]),
        None,
        x_scale=np.array([2.0, 0.5]),
        max_nfev=8,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.stack([ours.value["a"], ours.value["b"][0]]), theirs.x, rtol=1e-10
    )


def test_an_unknown_x_scale_is_refused():
    # Under runtime type checking (as in this test suite) the annotation refuses it
    # first, with a `TypeError`; without, `init` does, with a `ValueError`.
    with pytest.raises((ValueError, TypeError), match="x_scale"):
        optx.least_squares(
            _rosenbrock, optx.TRF(1e-8, 1e-8, x_scale="auto"), jnp.array([0.0, 0.0])
        )
