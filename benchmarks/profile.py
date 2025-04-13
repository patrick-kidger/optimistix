import json
import os
import pathlib
import re
import warnings
from typing import Literal

import fire
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt


def find_benchmark_directory():
    """Find the .benchmarks directory in the current or parent directory."""
    script_dir = pathlib.Path(__file__).parent
    benchmarks_dir = script_dir / ".benchmarks"

    if not benchmarks_dir.exists():
        benchmarks_dir = script_dir.parent / ".benchmarks"

    if not benchmarks_dir.exists():
        raise FileNotFoundError(
            "Error: Could not find .benchmarks directory. "
            "Please ensure it exists in the current or parent directory."
        )

    print(f"Using benchmarks from: {benchmarks_dir}")
    return benchmarks_dir


def find_benchmark_run(
    benchmarks_dir: pathlib.Path,
    platform: str,
    python_version: str,
    precision: str,
    iD: str,
) -> dict:
    """Find the benchmark run data based on the provided arguments.

    **Arguments:**

    - `benchmarks_dir`: The directory where the benchmarks are stored.
    - `platform`: The platform name (e.g., "Darwin").
    - `python_version`: The Python version (e.g., "3.12").
    - `precision`: The bit precision (e.g., "64bit").
    - `iD`: A 4-digit identifier for the benchmark run.

    **Returns:**

    - A dictionary containing the benchmark data loaded from the JSON file.
    """

    if platform not in ["Darwin"]:  # TODO: add Linux and Windows
        raise ValueError(
            f"Error: Platform '{platform}' is not supported. "
            "Currently, only Darwin is supported."
        )

    if re.match(r"^3\.([0-9]|1[0-3])\.", python_version):
        print(f"Valid Python version: {python_version}")
    else:
        print(f"Invalid Python version: {python_version}")

    if re.match(r"^(1|2|4|8|16|32|64|128)bit$", precision):
        print(f"Valid precision: {precision}")
    else:
        print(f"Invalid precision: {precision}")

    if not iD.isdigit() or len(iD) != 4:
        raise ValueError(f"Error: Run iD '{iD}' should be a 4-digit number.")

    folder_path = f"{platform}-CPython-{python_version}-{precision}"
    folder_path = benchmarks_dir / folder_path
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Error: Folder '{folder_path}' does not exist. "
            "Please check the platform, Python version, and precision."
        )

    json_files = [f for f in os.listdir(folder_path) if iD in f and f.endswith(".json")]
    assert len(json_files) == 1, (
        f"Error: Expected exactly one JSON file for run iD '{iD}' in "
        f"'{folder_path}', but found {len(json_files)} files."
    )
    [benchmark_run] = json_files
    benchmark_path = folder_path / benchmark_run

    with open(benchmark_path) as f:
        benchmark_data = json.load(f)

    if benchmark_data.get("commit_info", {}).get("dirty", False):
        warnings.warn(
            f"Benchmark file {benchmark_path} was run with uncommitted changes. "
            "Results may not be reproducible.",
            UserWarning,
        )

    return benchmark_data


def extract_solver_results(
    data: dict, solver_names: list, kind: Literal["runtime", "compile"]
) -> dict:
    """Extract results for specified solvers, grouped by problem. Returns a dictionary
    with the information required to generate performance profiles. The dictionary has
    the following structure:

    problems = {
        problem1: {
            solver1: {min_runtime: float, successful: bool},
            solver2: ...
        }
        problem2: ...
    }

    **Arguments:**

    - `data`: The benchmark data loaded from a JSON file.
    - `solver_names`: A list of solver names to extract results for.
    - `kind`: The kind of benchmark to extract results for, either "runtime" or
        "compile". This is used to filter the benchmarks in the data.

    **Returns:**

    - A dictionary containing the minimum runtime and success status for each solver on
        each problem.
    """
    print(f"Extracting results for solvers: {solver_names}")
    solver_data = {}

    for benchmark in data["benchmarks"]:
        if kind not in benchmark["name"]:  # Select either runtime or compile benchmarks
            continue
        else:
            solver_name = benchmark["extra_info"].get("solver name")
            problem_name = benchmark["extra_info"].get("problem name")
            min_runtime = benchmark["stats"]["min"]
            is_successful = benchmark["extra_info"].get("result", False)

            if solver_name in solver_names:
                if solver_name not in solver_data.keys():
                    solver_data[solver_name] = {}
                # TODO(jhaffner): currently assumes that problems are identifiable by
                # their names and ignores variably dimensioned problems, where several
                # problems would have the same name, but different attributes. We do not
                # yet support this in sif2jax, but support is planned for the short term
                solver_data[solver_name][problem_name] = {
                    "min_runtime": min_runtime,
                    "successful": is_successful,
                }

    return solver_data


def _solver_runtimes(comparable_data: dict) -> tuple[dict, jnp.ndarray]:
    comparable_runtimes = jnp.array(jtu.tree_leaves(comparable_data))
    if jnp.all(jnp.isnan(comparable_runtimes)):
        raise ValueError(
            "No problems were solved by any solver (all runtimes are NaN). "
            "Please check the solver data for correctness."
        )

    minimum_runtimes = jnp.nanmin(comparable_runtimes, axis=0)
    relative_runtimes = comparable_runtimes / minimum_runtimes

    unique_runtimes = jnp.unique(
        jnp.where(jnp.isfinite(relative_runtimes), relative_runtimes, 1.0)
    )  # Replace NaN with 1.0 for relative runtimes - 1.0 always exists
    return dict(zip(comparable_data.keys(), relative_runtimes)), unique_runtimes


def get_relative_performance(solver_data: dict) -> tuple[dict, jnp.ndarray]:
    """Process solver data + compute performance profiles. The performance profiles are
    returned as a pandas DataFrame that may be exported, and that can be used to
    generate a performance profile plot.

    The data frame has the relative runtimes in the first column and the fraction of
    problems solved within that runtime in the other columns, one for each solver.

    **Arguments:**

    - `solver_data`: A dictionary containing the solver data extracted from the
        benchmark data. The structure of this dictionary is described in the docstring
        of `extract_solver_results`.

    **Returns:**

    - A tuple containing:
        - A dictionary with solver names as keys and arrays of fractions of problems
            solved within each unique runtime as values.
        - An array of unique runtimes relative to the best solver (the x-axis in the
            performance profile plot).
    """
    solvers = solver_data.keys()
    if len(solvers) < 2:
        raise ValueError("At least two solvers are required for a comparison.")

    problem_sets = [set(data.keys()) for data in solver_data.values()]
    comparable_problems = set.intersection(*problem_sets)
    if len(comparable_problems) == 0:
        raise ValueError(
            "No comparable problems found: problems must have been attempted by all of "
            "the solvers to be compared."
        )

    comparable_data = {}
    for solver in solvers:
        runtimes = []
        successes = []
        for problem in comparable_problems:
            runtimes.append(solver_data[solver][problem]["min_runtime"])
            successes.append(solver_data[solver][problem]["successful"])

        runtimes = jnp.where(jnp.asarray(successes), jnp.asarray(runtimes), jnp.nan)
        comparable_data[solver] = runtimes

    relative_runtimes, unique_runtimes = _solver_runtimes(comparable_data)

    solved_fractions = {}
    for solver in solvers:
        fractions = []
        for runtime in unique_runtimes:
            # Count how many problems this solver solved within the given runtime
            count = (relative_runtimes[solver] <= runtime).sum()
            num_problems = relative_runtimes[solver].size
            fraction = count / num_problems
            fractions.append(fraction)
        solved_fractions[solver] = jnp.array(fractions)

    return solved_fractions, unique_runtimes


def plot_solver_performances(solver_performances: dict, unique_runtimes: jnp.ndarray):
    """Plot the performance of solvers based on the provided DataFrame."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for solver in solver_performances.keys():
        ax.plot(
            unique_runtimes,
            solver_performances[solver],
            label=solver,
            markersize=4,
            drawstyle="steps-post",
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)

    ax.set_xlabel("Relative Runtime (to best solver)")
    ax.set_ylabel("Fraction of Problems Solved")
    ax.set_title("Solver Performance Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()


def main(
    platform: str,
    python_version: str,
    precision: str,
    run_id: str,
    kind: Literal["runtime", "compile"],
    *solver_names: str,
):
    """Main entry point for the computation of solver performance profiles.

    **Arguments:**

    - `platform`: The operating system (e.g., "Darwin").
    - `python_version`: The Python version (e.g., "3.12").
    - `precision`: The bit precision (e.g., "64bit").
    - `run_id`: A 4-digit identifier for the benchmark run.
    - `kind`: The kind of benchmark to analyse, either "runtime" or "compile". Defaults
        to "runtime". We currently support these two flavours of benchmarks, and each
        solver is run at most once for each of these.
    - `solver_names`: Names of the solvers to compare. Optimistix solvers are prepended
        with "optx." (e.g., "optx.BFGS"), Scipy solvers are prepended with "scipy."
        At least two solvers must be provided for a comparison.
    """
    benchmarks_dir = find_benchmark_directory()
    benchmark_data = find_benchmark_run(
        benchmarks_dir, platform, str(python_version), precision, run_id
    )  # casting the python version to a string seems to be necessary with fire
    assert len(solver_names) > 1, "At least two solver names must be provided."

    solver_data = extract_solver_results(benchmark_data, [*solver_names], kind)
    solver_performances, unique_runtimes = get_relative_performance(solver_data)
    plot_solver_performances(solver_performances, unique_runtimes)


if __name__ == "__main__":
    fire.Fire(main)
