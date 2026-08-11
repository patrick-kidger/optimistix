import equinox as eqx
import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "strict")


@pytest.fixture(scope="module", autouse=True)
def clear_caches_between_modules():
    """Bound memory from compiled executables in the full parametrised test suite."""
    yield
    eqx.clear_caches()
    jax.clear_caches()


@pytest.fixture
def getkey():
    return eqxi.GetKey()
