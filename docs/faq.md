# FAQ

## How does this differ from JAXopt?

**Reasons to use Optimistix:**

- Optimistix is usually much faster to compile, and faster to run.
- Optimistix support several solvers not found in JAXopt (e.g. [`optimistix.Newton`][] for root-finding problems).
- Optimistix's APIs are designed to integrate more cleanly with the scientific ecosystem being built up around [Equinox](https://github.com/patrick-kidger/equinox).
- Optimistix is much more flexible for advanced use-cases, see e.g. the way we can [mix-and-match](./mix-and-match.md) different optimisers.

**Reasons to use JAXopt:**

- JAXopt supports optimising functions that can't be `jax.jit`'d. (It is good practice to always `jit` your functions for speed though! Nonetheless this is an edge case they support that we don't.)
- JAXopt has better support for (a) constrained optimisation and (b) quadratic programming. Right now Optimistix only supports hypercube constraints in its root finding algorithms, and quadratic programming is out-of-scope.

## How does this differ from `jax.scipy.optimize.minimize`?

This is an API which is likely to be removed from JAX in the near future, in favour of Optimistix and JAXopt. (The core JAX API only supports minimisation, and only supports the BFGS algorithm.)
