# FAQ

## How does Optimistix compare against...

#### ...Optax?

Optax is specifically dedicated to first-order gradient based minimisation methods. (Which are the class of algorithms that are most useful for training neural networks.) Optimistix has a much broader scope: it also includes other kinds of algorithms (e.g. [`optimistix.LevenbergMarquardt`][] or [`optimistix.Newton`][]), and other kinds of problems (e.g. root finding).

Optimistix doesn't try to reinvent the wheel! The Optax library is excellent. As such Optimistix deliberately has relatively few first-order gradient methods, and you should usually pair Optimistix with Optax, via [`optimistix.OptaxMinimiser`][].

#### ...JAXopt?

**Reasons to use Optimistix:**

- Optimistix is typically faster to compile, and faster to run.
- Optimistix supports some solvers not found in JAXopt (e.g. [`optimistix.Newton`][] for root-finding problems).
- Optimistix's APIs are designed to integrate more cleanly with the scientific ecosystem being built up around [Equinox](https://github.com/patrick-kidger/equinox).
- Optimistix is much more flexible for advanced use-cases, see e.g. the way we can [mix-and-match](./api/searches/introduction.md) different optimisers.

**Reasons to use JAXopt:**

- JAXopt supports optimising functions that can't be `jax.jit`'d. (It is good practice to always `jit` your functions, for speed. Nonetheless this is an edge case JAXopt supports that Optimistix doesn't.)
- JAXopt supports (a) constrained optimisation and (b) quadratic programming. Right now Optimistix (a) only supports hypercube constraints and only in its root finding algorithms, and (b) quadratic programming is out-of-scope.

#### ...`jax.scipy.optimize.minimize`?

This is an API which is likely to be removed from JAX at some point, in favour of Optimistix and JAXopt. Don't use it. (Note that the core JAX API only supports minimisation, and only supports the BFGS algorithm.)

## How to debug a solver that is failing to converge, or producing an error?

This is an unfortunately common occurence! Nonlinear optimisation is a difficult problem, so no solver is guaranteed to converge. Optimistix prefers to explicitly raise an error rather than silently return a suboptimal result.

1. First of all, many of the same standard debugging advice for any JAX program applies. See [this guidance in the Equinox documentation](https://docs.kidger.site/equinox/api/debug/) for a list of standard debugging tools, how to handle NaNs, etc.

2. Likewise, if you are getting a runtime error message from Optimistix (`XlaRuntimeError: ...`), then setting the `EQX_ON_ERROR=breakpoint` environment variable is usually the most useful place to start. [The documentation for `eqx.error_if`](https://docs.kidger.site/equinox/api/errors/#equinox.error_if) discusses runtime errors further.

3. If you are happy with a suboptimal result, and just want to move on with your computation, then you can pass `throw=False` to `optimistix.{minimise, least_squares, root_find, fixed_point}` to ignore the error.

4. You can try other solvers; see [how to choose a solver](./how-to-choose.md).

5. Some solvers provide a `verbose` flag, e.g. `optimistix.LevenbergMarquardt(..., verbose=...)`, which will print out some information about the state of the solve.

6. If you are getting a solution, but it is worse then you are expecting, then the solver may have converged to a local minima. For this, changing the target problem is usually the best approach. (For example when fitting a time series: don't try to fit the whole time series in one go. Instead only fit the start of the time series, and when the model is better-trained later on, then start lengthening how much of the time series you feed into the loss.)

7. Finally, if all else fails: start placing down `jax.debug.print` and `jax.debug.breakpoint` statements, and start bisecting through the internals of the solver. Adding debug statements to the Optimistix source code (located at `import optimistix; print(optimistix.__file__)`) is always going to be the most powerful approach. (This is really standard advice for working with any library, but it needs restating surprisingly often! See also [this standard advice for unsticking yourself](https://kidger.site/thoughts/how-to-handle-a-hands-off-supervisor/#unsticking-yourself).)
