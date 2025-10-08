import functools as ft
import warnings

import equinox as eqx
import jax
import optax
import optimistix as optx
import pytest
import scipy as scp
import sif2jax

from .conftest import get_max_dimension, get_min_dimension


# Set a consistent number of maximum steps for all solvers. We might want to override
# this, make it solver-specific in the future, or make it a config option.
max_steps = 2**10

# Benchmark solvers that are part of documented API. Specifying the name is needed since
# it cannot be retrieved for the OptaxMinimisers.
unconstrained_minimisers = (
    (optx.BFGS(rtol=1e-3, atol=1e-6), "optx.BFGS"),
    (optx.LBFGS(rtol=1e-3, atol=1e-6, history_length=10), "optx.LBFGS"),
    (
        optx.OptaxMinimiser(
            optax.lbfgs(
                linesearch=optax.scale_by_backtracking_linesearch(
                    max_backtracking_steps=20
                )
            ),
            rtol=1e-3,
            atol=1e-6,
        ),
        "optax.lbfgs",
    ),
)
# Specify scipy minimisers using tuples (method_name: str, kwargs: dict)
unconstrained_scipy_minimisers = (
    ("BFGS", {}),
    (
        "L-BFGS-B",
        dict(
            max_corr=10,  # Corresponds to history_length in LBFGS
            ftol=1e-3,  # ftol corresponds to rtol, but only applied to f
        ),
    ),
)


def get_test_cases(problems):
    max_dimension = get_max_dimension()
    min_dimension = get_min_dimension()
    if max_dimension is not None:
        test_cases = []
        for problem in problems:
            problem_dimension = problem.y0().size
            if problem_dimension <= max_dimension:
                if min_dimension is not None:
                    if problem_dimension >= min_dimension:
                        test_cases.append(problem)
                else:
                    test_cases.append(problem)
        return tuple(test_cases)
    elif min_dimension is not None:
        test_cases = []
        for problem in problems:
            problem_dimension = problem.y0().size
            if problem_dimension >= min_dimension:
                test_cases.append(problem)
        return tuple(test_cases)
    else:
        return problems


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "problem", get_test_cases(sif2jax.unconstrained_minimisation_problems)
)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_runtime_unconstrained_minimisers(benchmark, minimiser, problem):
    solver, name = minimiser
    solve = eqx.filter_jit(
        ft.partial(
            optx.minimise,
            problem.objective,
            solver,
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

    if isinstance(solver, optx.OptaxMinimiser):

        def wrapped(y0):
            solution = solve(y0)
            objective_value = solution.state.f.block_until_ready()
            num_steps = solution.stats["num_steps"]
            return objective_value, solution.result, num_steps
    else:
        # For non-OptaxMinimiser solvers, we need to access the FunctionInfo
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
    benchmark.extra_info["problem dimension"] = int(problem.y0().size)
    benchmark.extra_info["solver name"] = name
    benchmark.extra_info["max steps"] = int(max_steps)


msg = (
    "This benchmark requires AOT compilation. To support this we either need to "
    "make all Optimistix solvers compatible with `jax.jit` compilation of our "
    "top-level APIs (`optx.{minimise, least_squares, ...}`) or support AOT "
    "compilation for functions compiled with `eqx.filter_jit`. The first of the "
    "two solutions is on the list for Optimistix. Supporting it requires "
    "different handling of solvers that carry a jaxpr in their state. "
)


@pytest.mark.skip(reason=msg)
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "problem", get_test_cases(sif2jax.unconstrained_minimisation_problems)
)
@pytest.mark.parametrize("minimiser", unconstrained_minimisers)
def test_compilation_time_unconstrained_minimisers(benchmark, minimiser, problem):
    solver, name = minimiser

    def wrapped(y0):
        def _solve(y0):
            return optx.minimise(problem.objective, solver, y0, problem.args())

        return jax.jit(lambda x: _solve(x)).lower(y0).compile()

    # Benchmark the runtime of the compiled function
    wrapped(problem.y0())  # Warm up
    benchmark(wrapped, problem.y0())

    # Save information and results
    benchmark.extra_info["problem name"] = problem.__class__.__name__
    benchmark.extra_info["solver name"] = name


@pytest.mark.skipif("not config.getoption('scipy')")
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
    benchmark.extra_info["problem dimension"] = int(problem.y0().size)
    benchmark.extra_info["solver name"] = "scipy." + method
    benchmark.extra_info["max steps"] = int(max_steps)
