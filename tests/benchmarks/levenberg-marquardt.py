# Optimistix vs JAXopt for a small nonlinear least-squares problem.
#
# Each pairing of (nonlinear library, linear solver) is shown.
# We make exactly one step of the solver, to normalise across differences for how the
# trust region radius is handled, or differing termination conditions. This is performed
# on a laptop CPU, with Optimistix version 0.0.1 and JAXopt version 0.7.
#
# --------------------------------------------------------------------------------------
#
# Levenberg--Marquardt timing: taking exactly one step.
# ---------
# Optimistix (QR; default):
# Compile+run time: 8.616801428001054
# Run time: 0.002039344999502646
# ----
# Optimistix (normal CG):
# Compile+run time: 25.256638882001425
# Run time: 0.015216022002277896
# ----
# Optimistix (normal Cholesky):
# Compile+run time: 7.744616070001939
# Run time: 0.0015822620007384103
# ----
# JAXopt (normal CG; default):
# Compile+run time: 60.017054975000065
# Run time: 0.018481929000699893
# ----
# JAXopt (normal Cholesky):
# Compile+run time: 10.43396132700218
# Run time: 0.0017714580026222393

# --------------------------------------------------------------------------------------
#
# In conclusion, we see that Optimistix is consistently faster than JAXopt, at both
# compile time and runtime. It also has a more useful default.

import functools as ft
import timeit

import diffrax as dfx  # pyright: ignore
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt  # pyright: ignore
import lineax as lx
import optimistix as optx
from jaxtyping import Array, Float


def vector_field(
    t, y: Float[Array, "2"], parameters: Float[Array, "4"]
) -> Float[Array, "2"]:
    """Lotka--Volterra equations."""
    prey, predator = y
    α, β, γ, δ = parameters
    d_prey = α * prey - β * prey * predator
    d_predator = -γ * predator + δ * prey * predator
    d_y = jnp.stack([d_prey, d_predator])
    return d_y


def solve(
    parameters: Float[Array, "4"], y0: Float[Array, "2"], saveat: dfx.SaveAt
) -> Float[Array, " ts"]:
    """Solve a single ODE."""
    term = dfx.ODETerm(vector_field)
    solver = dfx.Tsit5()
    t0 = saveat.subs.ts[0]
    t1 = saveat.subs.ts[-1]
    dt0 = 0.1
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args=parameters,
        saveat=saveat,
        # support forward-mode autodiff, which is used by Levenberg--Marquardt
        adjoint=dfx.DirectAdjoint(),
    )
    return sol.ys  # pyright: ignore


def get_data() -> tuple[Float[Array, "3 2"], Float[Array, "3 50"]]:
    """Simulate some training data."""
    # We consider three possible initial conditions.
    y0_a = jnp.array([9.0, 9.0])
    y0_b = jnp.array([10.0, 10.0])
    y0_c = jnp.array([11.0, 11.0])
    y0s = jnp.stack([y0_a, y0_b, y0_c])
    true_parameters = jnp.array([0.1, 0.02, 0.4, 0.02])
    saveat = dfx.SaveAt(ts=jnp.linspace(0, 30, 20))
    batch_solve = eqx.filter_jit(eqx.filter_vmap(solve, in_axes=(None, 0, None)))
    values = batch_solve(true_parameters, y0s, saveat)
    return y0s, values


def residuals(parameters, y0s__values):
    """The residuals, that we would like to minimise as a nonlinear least square."""
    y0s, values = y0s__values
    saveat = dfx.SaveAt(ts=jnp.linspace(0, 30, 20))
    batch_solve = eqx.filter_vmap(solve, in_axes=(None, 0, None))
    pred_values = batch_solve(parameters, y0s, saveat)
    return values - pred_values


# Lineax deliberately doesn't offer this as a solver.
# It's mostly just a worse version of QR. It runs at similar speed empirically (see the
# benchmarks below), but is substantially less accurate.
# Anyway, we have an implementation here to provide a fair comparison with JAXopt, which
# uses it as its default solver.
class NormalCholesky(lx.AbstractLinearSolver):
    def init(self, operator, options):
        del options
        matrix = operator.as_matrix()
        factor, lower = jsp.linalg.cho_factor(matrix.T @ matrix)
        assert lower is False
        packed_structures = lx.internal.pack_structures(operator)
        return matrix, factor, packed_structures

    def compute(self, state, vector, options):
        matrix, factor, packed_structures = state
        vector = lx.internal.ravel_vector(vector, packed_structures)
        solution = jsp.linalg.cho_solve((factor, False), matrix.T @ vector)
        return (
            lx.internal.unravel_solution(solution, packed_structures),
            lx.RESULTS.successful,
            {},
        )

    def transpose(self, state, options):
        assert False

    def allow_dependent_columns(self, operator):
        assert False

    def allow_dependent_rows(self, operator):
        assert False


# Default option for Optimistix. (QR linear solver.)
@jax.jit
def optx_qr(init_parameters, y0s, values):
    optx_solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(
        residuals,
        optx_solver,
        init_parameters,
        args=(y0s, values),
        max_steps=1,
        throw=False,
    )
    return sol.value


# Normal-CG linear solver. Chosen to match the JAXopt default below.
@jax.jit
def optx_normal_cg(init_parameters, y0s, values):
    # Explicitly set the number of iterations, to be sure we have the same number of
    # iterations between Optimistix and JAXopt.
    cg = lx.NormalCG(atol=0, rtol=0, max_steps=10)
    optx_solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8, linear_solver=cg)
    sol = optx.least_squares(
        residuals,
        optx_solver,
        init_parameters,
        args=(y0s, values),
        max_steps=1,
        throw=False,
    )
    return sol.value


# Normal-Cholesky linear solver. Chosen to match the fastest JAXopt option below (if you
# know what you're doing).
@jax.jit
def optx_normal_cholesky(init_parameters, y0s, values):
    optx_solver = optx.LevenbergMarquardt(
        rtol=1e-8, atol=1e-8, linear_solver=NormalCholesky()
    )
    sol = optx.least_squares(
        residuals,
        optx_solver,
        init_parameters,
        args=(y0s, values),
        max_steps=1,
        throw=False,
    )
    return sol.value


# Default options for JAXopt. (Nonlinear-CG linear solver.)
@jax.jit
def jaxopt_normal_cg(init_parameters, y0s, values):
    # JAXopt doesn't take extra parameters. Also, JAXopt requires a single vector as the
    # output -- multidimensional arrays and pytrees are not supported.
    def residual_fn(parameters):
        return residuals(parameters, (y0s, values)).reshape(-1)

    # Explicitly set the number of iterations, to be sure we have the same number of
    # iterations between Optimistix and JAXopt.
    cg = ft.partial(jaxopt.linear_solve.solve_cg, tol=0, maxiter=10)
    jaxopt_solver = jaxopt.LevenbergMarquardt(residual_fn, maxiter=1, solver=cg)
    return jaxopt_solver.run(init_parameters).params


# Normal-Cholesky linear solver. This is as fast as JAXopt seems to go. (If you know
# what you're doing.)
@jax.jit
def jaxopt_normal_cholesky(init_parameters, y0s, values):
    def residual_fn(parameters):
        return residuals(parameters, (y0s, values)).reshape(-1)

    jaxopt_solver = jaxopt.LevenbergMarquardt(
        residual_fn, solver="cholesky", materialize_jac=True, maxiter=1
    )
    return jaxopt_solver.run(init_parameters).params


init_parameters = jnp.zeros(4)
(y0s, values) = get_data()

run_optx_qr = lambda: optx_qr(init_parameters, y0s, values).block_until_ready()
run_optx_normal_cg = lambda: optx_normal_cg(
    init_parameters, y0s, values
).block_until_ready()
run_optx_normal_cholesky = lambda: optx_normal_cholesky(
    init_parameters, y0s, values
).block_until_ready()
run_jaxopt_normal_cg = lambda: jaxopt_normal_cg(
    init_parameters, y0s, values
).block_until_ready()
run_jaxopt_normal_cholesky = lambda: jaxopt_normal_cholesky(
    init_parameters, y0s, values
).block_until_ready()

optx_qr_compile_time = timeit.timeit(run_optx_qr, number=1)
optx_normal_cg_compile_time = timeit.timeit(run_optx_normal_cg, number=1)
optx_normal_cholesky_compile_time = timeit.timeit(run_optx_normal_cholesky, number=1)
jaxopt_normal_cg_compile_time = timeit.timeit(run_jaxopt_normal_cg, number=1)
jaxopt_normal_cholesky_compile_time = timeit.timeit(
    run_jaxopt_normal_cholesky, number=1
)

optx_qr_run_time = min(timeit.repeat(run_optx_qr, number=1, repeat=10))
optx_normal_cg_run_time = min(timeit.repeat(run_optx_normal_cg, number=1, repeat=10))
optx_normal_cholesky_run_time = min(
    timeit.repeat(run_optx_normal_cholesky, number=1, repeat=10)
)
jaxopt_normal_cg_run_time = min(
    timeit.repeat(run_jaxopt_normal_cg, number=1, repeat=10)
)
jaxopt_normal_cholesky_run_time = min(
    timeit.repeat(run_jaxopt_normal_cholesky, number=1, repeat=10)
)

# We do exactly one step so that we don't need to worry about their differing
# terimination conditions and other differences in the algorithm.
print("Levenberg--Marquardt timing: taking exactly one step.")
print("---------")
print("Optimistix (QR; default):")
print(f"Compile+run time: {optx_qr_compile_time}")
print(f"Run time: {optx_qr_run_time}")
print("----")
print("Optimistix (normal CG):")
print(f"Compile+run time: {optx_normal_cg_compile_time}")
print(f"Run time: {optx_normal_cg_run_time}")
print("----")
print("Optimistix (normal Cholesky):")
print(f"Compile+run time: {optx_normal_cholesky_compile_time}")
print(f"Run time: {optx_normal_cholesky_run_time}")
print("----")
print("JAXopt (normal CG; default):")
print(f"Compile+run time: {jaxopt_normal_cg_compile_time}")
print(f"Run time: {jaxopt_normal_cg_run_time}")
print("----")
print("JAXopt (normal Cholesky):")
print(f"Compile+run time: {jaxopt_normal_cholesky_compile_time}")
print(f"Run time: {jaxopt_normal_cholesky_run_time}")
