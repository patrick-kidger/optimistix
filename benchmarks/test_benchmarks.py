# Running benchmarks to test the performance of our solvers, using the pytest
# benchmarking extension.
#
# To run the benchmarks, simply use `pytest benchmarks` from the root directory.
# If you'd like to save the results, the flag `--benchmark-save=<file_path>` can be used
# to save a .json file with stats and additional custom metrics.

# open questions: should benchmarks be version controlled? This would be necessary to
# compare performance across versions, but I'm not sure how large the files would be.


import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
import pytest


def block_tree_until_ready(x):
    dynamic, static = eqx.partition(x, eqx.is_inexact_array)
    dynamic = jtu.tree_map(lambda x: x.block_until_ready(), dynamic)
    return eqx.combine(dynamic, static)


# Dummy starting point - benchmark only solvers that are part of documented library, not
# custom compbinations
unconstrained_minimisers = (optx.BFGS(rtol=1e-3, atol=1e-6),)

# Dummy problem to get things started - should import from CUTEST collection
unconstrained_problems = (
    # fn, y0, args, expected_result
    (
        lambda y, args: jnp.sum((y - 1) ** 2),
        2 * jnp.ones(10),
        None,
        jnp.ones(10),
    ),
)


@pytest.mark.benchmark
@pytest.mark.parametrize("fn, y0, args, expected_result", unconstrained_problems)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_benchmark_unconstrained_minimisers(
    benchmark, minimiser, fn, y0, args, expected_result
):
    compiled = eqx.filter_jit(eqx.Partial(optx.minimise, fn, minimiser, y0, args))

    def wrapped():
        return block_tree_until_ready(compiled())  # Returns an optx.Solution

    _ = wrapped()

    # Benchmark the runtime of the compiled function
    result = benchmark.pedantic(wrapped, rounds=5, iterations=1)
    assert isinstance(result, optx.Solution)
