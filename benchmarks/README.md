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