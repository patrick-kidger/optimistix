import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_numpy_dtype_promotion", "strict")


@pytest.fixture
def getkey():
    return eqxi.GetKey()


# TODO: this can probably be done in a more streamlined fashion!
# Skip benchmarks on default test runs. Request these explicitly (e.g. when stress-
# testing new features).
def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-benchmarks"):
        skip_benchmarks = pytest.mark.skip(reason="Skip unless --run-benchmarks used.")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmarks)


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmarks", action="store_true", default=False, help="run benchmarks"
    )
