import functools as ft

import jax
import optimistix as optx
import pytest

from .helpers import cutest_unconstrained_problems


cutest = pytest.mark.skipif("not config.getoption('cutest')")


# Benchmark solvers that are part of documented API.
unconstrained_minimisers = (optx.BFGS(rtol=1e-3, atol=1e-6),)


@cutest
@pytest.mark.benchmark
@pytest.mark.parametrize("problem", cutest_unconstrained_problems)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_runtime_unconstrained_minimisers(benchmark, minimiser, problem):
    solve = jax.jit(
        ft.partial(
            optx.minimise,
            problem.objective,
            minimiser,
            args=problem.args(),
            max_steps=2**8,  # TODO: increase number of steps - for tough problems
            throw=False,
        )
    )

    _ = solve(problem.y0())  # Warm up

    def wrapped(y0):
        solution = solve(y0)
        objective_value = solution.state.f_info.f.block_until_ready()
        num_steps = solution.stats["num_steps"]
        return objective_value, solution.result, num_steps

    # Benchmark the runtime of the compiled function
    values = benchmark.pedantic(wrapped, args=(problem.y0(),), rounds=5, iterations=1)

    # Save information and results
    objective_value, result, num_steps = values
    benchmark.extra_info["number of steps"] = num_steps
    benchmark.extra_info["objective value"] = objective_value
    benchmark.extra_info["result"] = result
    benchmark.extra_info["problem name"] = problem.__class__.__name__
    benchmark.extra_info["solver name"] = minimiser.__class__.__name__
