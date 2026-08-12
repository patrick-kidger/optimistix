import jax
import jax.numpy as jnp
import optimistix as optx


def _quadratic(y, args):
    del args
    return jnp.sum(y**2)


def _solve(progress_meter, *, max_steps=64):
    solver = optx.BFGS(rtol=1e-6, atol=1e-6)
    y0 = jnp.array([2.0, -3.0])
    return optx.minimise(
        _quadratic,
        solver,
        y0,
        max_steps=max_steps,
        progress_meter=progress_meter,
        throw=False,
    )


def test_no_progress_meter(capfd):
    capfd.readouterr()
    _solve(optx.NoProgressMeter())
    jax.effects_barrier()
    captured = capfd.readouterr()
    assert captured.out == ""


def test_text_progress_meter(capfd):
    capfd.readouterr()
    _solve(optx.TextProgressMeter(minimum_increase=0.0))
    jax.effects_barrier()
    captured = capfd.readouterr()
    assert captured.out.startswith("0.00%")
    assert captured.out.endswith("100.00%\n")


def test_tqdm_progress_meter(capfd):
    capfd.readouterr()
    _solve(optx.TqdmProgressMeter(refresh_steps=1))
    jax.effects_barrier()
    captured = capfd.readouterr()
    assert captured.err.count("\r") >= 1
    assert "100.00%" in captured.err.rsplit("\r", 1)[-1]


def test_tqdm_progress_meter_jit(capfd):
    capfd.readouterr()
    jax.jit(lambda: _solve(optx.TqdmProgressMeter(refresh_steps=1)))()
    jax.effects_barrier()
    captured = capfd.readouterr()
    assert captured.err.count("\r") >= 1
    assert "100.00%" in captured.err.rsplit("\r", 1)[-1]
