# In this comparison, we compare Optimistix against JAXopt for backpropagating through
# a vmap'd unrolled loop. JAXopt has serious performance issues in this case: its
# runtime grows with the maximum number of steps, even if these steps aren't actually
# used.
#
# Here's some typical printout (laptop CPU). Incidentally if we take it a step further,
# to a million steps, then JAXopt just crashes with an out-of-memory instead.
#
# Comparison of vmap + unrolled autodiff.
# ------------------------
# Optimistix:
# max_steps=10000 runtime=0.01200
# max_steps=100000 runtime=0.01127
# ---------
# JAXopt:
# max_steps=10000 runtime=0.29483
# max_steps=100000 runtime=3.15099

import timeit

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt  # pyright: ignore
import optimistix as optx


A_sqrt = jr.normal(jr.PRNGKey(0), (100, 100))
A = A_sqrt.T @ A_sqrt  # now positive definite
x0 = jr.normal(jr.PRNGKey(1), (1, 100))
optx_times = {}
jaxopt_times = {}

for max_steps in (10_000, 100_000):

    @jax.jit
    @jax.vmap
    @jax.grad
    def run_optx(x0):
        def fn(x, _):
            return x.T @ A @ x

        solver = optx.GradientDescent(learning_rate=1e-3, rtol=1e-4, atol=1e-4)
        sol = optx.minimise(
            fn,
            solver,
            x0,
            adjoint=optx.RecursiveCheckpointAdjoint(),
            max_steps=max_steps,
        )
        return jnp.sum(sol.value)

    optx_times[max_steps] = min(
        timeit.repeat(lambda: run_optx(x0), number=1, repeat=10)
    )

    @jax.jit
    @jax.vmap
    @jax.grad
    def run_jaxopt(x0):
        def fn(x):
            return x.T @ A @ x

        solver = jaxopt.GradientDescent(
            fn,
            stepsize=1e-3,
            tol=1e-4,
            acceleration=False,
            maxiter=max_steps,
            implicit_diff=False,
        )
        return jnp.sum(solver.run(x0).params)

    jaxopt_times[max_steps] = min(
        timeit.repeat(lambda: run_jaxopt(x0), number=1, repeat=10)
    )

print("Comparison of vmap + unrolled autodiff.")
print("------------------------")
print("Optimistix:")
for k, v in optx_times.items():
    print(f"max_steps={k} runtime={v:.5f}")
print("---------")
print("JAXopt:")
for k, v in jaxopt_times.items():
    print(f"max_steps={k} runtime={v:.5f}")
