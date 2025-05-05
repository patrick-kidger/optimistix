import functools as ft

import equinox as eqx
import jax
import jax.tree_util as jtu
import optimistix as optx
import pytest
import sif2jax


# TODO: perhaps restrict JAX use of threads (certainly for comparison against eg. scipy)


cutest = pytest.mark.skipif("not config.getoption('cutest')")


# Benchmark solvers that are part of documented API.
unconstrained_minimisers = (optx.BFGS(rtol=1e-3, atol=1e-6),)

@cutest
@pytest.mark.benchmark
@pytest.mark.parametrize("problem", sif2jax.unconstrained_minimisation_problems)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_runtime_unconstrained_minimisers(benchmark, minimiser, problem):
    solve = jax.jit(
        ft.partial(
            optx.minimise,
            problem.objective,
            minimiser,
            problem.y0(),
            problem.args(),
            max_steps=2**10,  # TODO troubleshooting
            # throw=False,
        )
    )

    _ = solve(problem.y0())  # Warm up

    def wrapped(y0):
        solution = solve(y0)
        objective_value = solution.state.f_info.f.block_until_ready()
        num_steps = solution.stats["num_steps"]
        return objective_value, solution.result, num_steps

    values = benchmark.pedantic(wrapped, args=(problem.y0(),), rounds=5, iterations=1)

    # Save information and results
    objective_value, result, num_steps = values
    benchmark.extra_info["number of steps"] = num_steps
    benchmark.extra_info["objective value"] = objective_value
    benchmark.extra_info["result"] = result
    benchmark.extra_info["problem name"] = problem.__class__.__name__
    benchmark.extra_info["solver name"] = minimiser.__class__.__name__


constrained_minimisers = (optx.IPOPTLike(rtol=1e-2, atol=1e-2),)


# @pytest.mark.benchmark
# @pytest.mark.parametrize("problem", constrained_problems)
# @pytest.mark.parametrize("minimiser", constrained_minimisers)
# def test_benchmark_constrained_minimisers(benchmark, minimiser, problem):
#     compiled = eqx.filter_jit(
#         eqx.Partial(
#             optx.minimise,
#             problem.objective,
#             minimiser,
#             problem.y0(),
#             problem.args(),
#             constraint=problem.constraint,
#             bounds=problem.bounds(),
#         )
#     )

#     def wrapped():
#         return block_tree_until_ready(compiled())  # Returns an optx.Solution

#     _ = wrapped()  # Warm up

#     # Benchmark the runtime of the compiled function
#     result = benchmark.pedantic(wrapped, rounds=5, iterations=1)
#     benchmark.extra_info["number of steps"] = result.stats["num_steps"]
