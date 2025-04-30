import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "standard")
# Remark Peter: limit JAX to single-thread with taskset -c 0 pytest --cutest


@pytest.fixture
def getkey():
    return eqxi.GetKey()


def pytest_addoption(parser):
    parser.addoption(
        "--cutest",
        action="store_true",
        dest="cutest",
        default=False,
        help="Enable benchmark (CUTEST) tests.",
    )
