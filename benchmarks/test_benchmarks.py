import functools as ft
import warnings

import equinox as eqx
import jax
import optimistix as optx
import pytest
import scipy as scp
import sif2jax

from .conftest import get_max_dimension


scipy = pytest.mark.skipif("not config.getoption('scipy')")


# Set a consistent number of maximum steps for all solvers. We might want to override
# this, make it solver-specific in the future, or make it a config option.
max_steps = 2**8

# Benchmark solvers that are part of documented API.
unconstrained_minimisers = (optx.BFGS(rtol=1e-3, atol=1e-6),)
# Specify scipy minimisers using tuples (method_name: str, kwargs: dict)
unconstrained_scipy_minimisers = (("BFGS", {}),)  # TODO(jhaffner): scipy tolerances?


def get_test_cases(problems):
    max_dimension = get_max_dimension()
    if max_dimension is not None:
        test_cases = []
        for problem in problems:
            problem_dimension = problem.y0().size
            if problem_dimension <= max_dimension:
                test_cases.append(problem)
        return tuple(test_cases)
    else:
        return problems


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "problem", get_test_cases(sif2jax.unconstrained_minimisation_problems)
)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_runtime_unconstrained_minimisers(benchmark, monkeypatch, minimiser, problem):
    monkeypatch.setenv("EQX_ON_ERROR", "nan")
    solve = jax.jit(
        ft.partial(
            optx.minimise,
            problem.objective,
            minimiser,
            args=problem.args(),
            max_steps=max_steps,
            # We save on failure and filter during analysis to be able to compute the
            # fraction of solved problems for performance profiles.
            throw=False,
        )
    )
    y0 = jax.tree.map(
        lambda x: jax.device_put(x) if eqx.is_array(x) else x, problem.y0()
    )

    solve(y0)  # Warm up

    def wrapped(y0):
        solution = solve(y0)
        objective_value = solution.state.f_info.f.block_until_ready()
        num_steps = solution.stats["num_steps"]
        return objective_value, solution.result, num_steps

    # Benchmark the runtime of the compiled function
    values = benchmark(wrapped, problem.y0())

    # Save information and results, convert to Python types for serialisation
    objective_value, result, num_steps = values
    benchmark.extra_info["number of steps"] = int(num_steps)
    benchmark.extra_info["objective value"] = float(objective_value)
    benchmark.extra_info["result"] = bool(result == optx.RESULTS.successful)
    benchmark.extra_info["problem name"] = problem.__class__.__name__
    benchmark.extra_info["solver name"] = "optx." + minimiser.__class__.__name__


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "problem", get_test_cases(sif2jax.unconstrained_minimisation_problems)
)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_compilation_time_unconstrained_minimisers(benchmark, minimiser, problem):
    def wrapped(y0):
        def _solve(y0):
            return optx.minimise(problem.objective, minimiser, y0, problem.args())

        return jax.jit(_solve).lower(y0).compile()

    # Benchmark the runtime of the compiled function
    wrapped(problem.y0())  # Warm up
    benchmark(wrapped, problem.y0())

    # Save information and results
    benchmark.extra_info["problem name"] = problem.__class__.__name__
    benchmark.extra_info["solver name"] = "optx." + minimiser.__class__.__name__


@scipy
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "problem", get_test_cases(sif2jax.unconstrained_minimisation_problems)
)
@pytest.mark.parametrize("minimiser", unconstrained_scipy_minimisers)
def test_runtime_unconstrained_scipy_minimisers(benchmark, minimiser, problem):
    # Provide a jitted objective function, gradient and Hessian
    objective = jax.jit(problem.objective)
    _ = objective(problem.y0(), problem.args())  # Warm up
    gradient = jax.jit(jax.grad(problem.objective))
    _ = gradient(problem.y0(), problem.args())
    hessian = jax.jit(jax.hessian(problem.objective))
    _ = hessian(problem.y0(), problem.args())

    method, kwargs = minimiser

    # Ensure that we set a consistent default of maximum iterations, but allow solver-
    # specific overrides in principle.
    options = kwargs.get("options", {})
    if "maxiter" not in options:
        options["maxiter"] = max_steps
    kwargs["options"] = options

    def wrapped(y0):
        # Call the scipy minimiser
        solution = scp.optimize.minimize(
            objective,
            y0,
            args=problem.args(),
            method=method,
            jac=gradient,
            hess=hessian,
            **kwargs,
        )
        return solution.fun, solution.success, solution.nit

    # Benchmark the runtime of scipy optimise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore warnings from scipy.optimize.minimize - they break benchmark saving
        values = benchmark(wrapped, problem.y0())

    # Save information
    objective_value, result, num_steps = values
    benchmark.extra_info["number of steps"] = int(num_steps)
    benchmark.extra_info["objective value"] = float(objective_value)
    benchmark.extra_info["result"] = bool(result)
    benchmark.extra_info["problem name"] = problem.__class__.__name__
    benchmark.extra_info["solver name"] = "scipy." + method
