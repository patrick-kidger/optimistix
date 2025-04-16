# Running benchmarks to test the performance of our solvers, using the pytest
# benchmarking extension.
#
# To run the benchmarks, simply run `pytest benchmarks` from the root directory.
# If you'd like to save the results, the flag `--benchmark-save=<file_path>` can be used
# to save a .json file with stats and additional custom metrics. (Useful to compare
# performance across versions.)


import equinox as eqx
import jax.tree_util as jtu
import optimistix as optx
import pytest

from .helpers import constrained_problems, unconstrained_problems


def block_tree_until_ready(x):
    dynamic, static = eqx.partition(x, eqx.is_inexact_array)
    dynamic = jtu.tree_map(lambda x: x.block_until_ready(), dynamic)
    return eqx.combine(dynamic, static)


# Benchmark solvers that are part of documented API.
unconstrained_minimisers = (optx.BFGS(rtol=1e-3, atol=1e-6),)
constrained_minimisers = (optx.IPOPTLike(rtol=1e-2, atol=1e-2),)


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


@pytest.mark.benchmark
@pytest.mark.parametrize("problem", constrained_problems)
@pytest.mark.parametrize("minimiser", constrained_minimisers)
def test_benchmark_constrained_minimisers(benchmark, minimiser, problem):
    compiled = eqx.filter_jit(
        eqx.Partial(
            optx.minimise,
            problem.objective,
            minimiser,
            problem.y0(),
            problem.args(),
            constraint=problem.constraint,
            bounds=problem.bounds(),
        )
    )

    def wrapped():
        return block_tree_until_ready(compiled())  # Returns an optx.Solution

    _ = wrapped()  # Warm up

    # Benchmark the runtime of the compiled function
    result = benchmark.pedantic(wrapped, rounds=5, iterations=1)
    benchmark.extra_info["number of steps"] = result.stats["num_steps"]
