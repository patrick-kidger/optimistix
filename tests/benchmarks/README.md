# Running benchmarks 

This folder contains a larger collection of tough problems. These are  useful to test 
performance of new solvers and new features, and to catch any regressions early.  

Benchmarks can be run with 

```
pytest --extensive
```

and will be skipped otherwise. With the option 

```
pytest --extensive --benchmark-save=<file_path>
```

benchmark results can be saved in a .json file. Additional custom metrics (e.g. the
number of iterations or the quality of the solution) and be configured in the benchmark
test functions. 

## Provenance of benchmark problems

CUTEST problems: sourced from https://bitbucket.org/optrove/sif/wiki/Home and parsed 
into a JAX-friendly format with the help of an LLM. They are then checked and compared
against known results, primary literature or other CUTEST interfaces by a human (yours 
truly) and corrected if necessary.