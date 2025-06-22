import os  # noqa: I001

os.environ["EQX_ON_ERROR"] = "nan"  # Make sure this is set before importing equinox
import equinox.internal as eqxi  # noqa: E402
import jax  # noqa: E402
import pytest  # noqa: E402


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "strict")


@pytest.fixture
def getkey():
    return eqxi.GetKey()


def pytest_addoption(parser):
    parser.addoption(
        "--scipy",
        action="store_true",
        dest="scipy",
        default=False,
        help="Enable comparison against scipy solvers on benchmark (CUTEST) tests.",
    )
    parser.addoption(
        "--max-dimension",
        action="store",
        type=int,
        default=None,
        help=(
            "Maximum dimension for optimization variables. "
            "Tests with higher dimensions will be skipped."
        ),
    )


def pytest_configure(config):
    global _max_dimension
    _max_dimension = config.getoption("--max-dimension")


def get_max_dimension():
    return _max_dimension
