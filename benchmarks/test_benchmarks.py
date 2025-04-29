import equinox as eqx
import jax.tree_util as jtu
import optimistix as optx
import pytest

from .cutest import unconstrained_problems


extensive = pytest.mark.skipif("not config.getoption('extensive')")


def block_tree_until_ready(x):
    dynamic, static = eqx.partition(x, eqx.is_inexact_array)
    dynamic = jtu.tree_map(lambda x: x.block_until_ready(), dynamic)
    return eqx.combine(dynamic, static)


# Benchmark solvers that are part of documented API.
unconstrained_minimisers = (optx.BFGS(rtol=1e-3, atol=1e-6),)


@extensive
@pytest.mark.benchmark
@pytest.mark.parametrize("fn, y0, args, expected_result", unconstrained_problems)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_benchmark_unconstrained_minimisers(
    benchmark, minimiser, fn, y0, args, expected_result
):
    compiled = eqx.filter_jit(eqx.Partial(optx.minimise, fn, minimiser, y0, args))

    def wrapped():
        return block_tree_until_ready(compiled())  # Returns an optx.Solution

    _ = wrapped()  # Warm up

    # Benchmark the runtime of the compiled function
    result = benchmark.pedantic(wrapped, rounds=5, iterations=1)
    benchmark.extra_info["number of steps"] = result.stats["num_steps"]
