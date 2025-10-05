# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

## Getting started

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/optimistix.git
cd optimistix
pip install -e .
```

Then install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

These hooks use ruff to format and lint the code, and pyright to typecheck it.

---

### If you're making changes to the code:**

Now make your changes. Make sure to include additional tests if necessary.

Next verify the tests all pass:

```bash
pip install -e '.[dev]'
pytest
```

If your changes could affect solver (or compilation) performance, please run the benchmark tests with 

```bash
pytest benchmarks/ --benchmark-only
```

You can run benchmarks on `main` / `dev` or on your feature branch for a before-and-after comparison, and also save more extensive results for analysis. For more on this, skip to the "Benchmarking" section below. 
Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

### If you're making changes to the documentation:

Make your changes. You can then build the documentation by doing

```bash
pip install -e '.[docs]'
mkdocs serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.


## Benchmarking

If you're interested in more extensive benchmarking - for instance when contributing a new solver - this section is for you. (Note that benchmarks are not run by default, and `--benchmark-only` is required to override this.)

You can save benchmark results with

```
pytest benchmarks/ --benchmark-save=<benchmark_name> --benchmark-only
```

and compare against previous runs with `pytest --benchmark-compare`, which will automatically pull in the last saved commit, but also takes run iDs as arguments. See the `pytest-benchmark` [documentation](https://pytest-benchmark.readthedocs.io/en/latest/usage.html#commandline-options) for more command line options. 
The `benchmark-autosave` option will specify the commit iD, instead of a user-defined name.
Make sure that you are running benchmarks with a clean working tree, so you can trace how changes affect performance!

For convenience, we support some custom flags: 

- `--min-dimension=<int>, --max-dimension=<int>` benchmarks can be run on a subset of problems based on problem size.
- `--scipy` benchmarks of our solvers are run against the corresponding Python implementation. You might want to limit problem dimension here, they can be quite slow.

pytest' `-k` flags also work in this setting to enable selective execution of benchmarking functions.

**Analysing benchmark results**

You can find a script to analyse benchmark results in `benchmarks/profile.py`. Run it with

```bash
python benchmarks/profile.py platform python_version precision iD kind *solver_names
```

Where platform refers to the platform on which the benchmarks were run (e.g. Darwin), precision is the numerical precision, e.g. 32bit, and iD is the benchmark run, a four-digit integer.
These are necessary to identify the saved results for the specific run. `kind` specifies if `runtime` or `compilation` benchmarks are to be compared, and solver names should be given as indicated in `benchmarks/test_benchmarks.py`.

**If you are contributing a solver**

In this case, you're probably reasonably familiar with the alternatives out there - if implementations we could compare to exist, please add them to the listed solvers in `benchmarks/test_benchmarks.py`, including hyperparameters such as solver tolerances to get as fair of a comparison as is feasible.
