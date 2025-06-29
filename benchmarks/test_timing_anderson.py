# Some tests to see the impact of the linear solver on the progression of
# the anderson solver.
from timeit import default_timer

import jax
import jaxtyping as jt
import lineax as lx
import optimistix as optx


@jax.jit
def test_function(x: jt.PyTree, *args):
    del args
    return {
        "a": (x["a"] - 1.0) * (jax.nn.tanh(x["b"][0]) - x["b"][1]),
        "b": (x["b"][0], jax.lax.cos(x["b"][1])),
    }


def time_solver(key: jt.PRNGKeyArray, lin_solver: lx.AbstractLinearSolver | None):
    solver: optx.AbstractFixedPointSolver
    if lin_solver is None:
        solver = optx.FixedPointIteration(1e-4, 1e-4)
    else:
        solver = optx.AndersonAcceleration(
            1e-4, 1e-4, mixing=0.5, linear_solver=lin_solver
        )

    @jax.jit
    @jax.vmap
    def solve(start: jt.PyTree) -> jt.PyTree:
        sol = optx.fixed_point(
            test_function, solver, start, throw=False, max_steps=10000
        )
        return sol.stats["num_steps"]

    y0 = {
        "a": jax.numpy.zeros((1000, 3, 4, 2)),
        "b": (
            jax.random.normal(key, (1000, 2)) / 10,
            jax.random.normal(key, (1000, 2)) / 10,
        ),
    }
    solve(y0)

    t0 = default_timer()
    stats = solve(y0)
    dt = default_timer() - t0

    stps = stats.mean().item()
    return dt / stps, dt, stps


if __name__ == "__main__":
    key = jax.random.key(666)
    print("FixedPointIteration:", time_solver(key, None))
    print("NCG:", time_solver(key, lx.NormalCG(1e-4, 1e-4)))
    print("BiCG:", time_solver(key, lx.BiCGStab(1e-4, 1e-4)))
    print("GMRES:", time_solver(key, lx.GMRES(1e-4, 1e-4)))
    print("SVD:", time_solver(key, lx.SVD()))
    print("LU:", time_solver(key, lx.LU()))
    print("QR:", time_solver(key, lx.QR()))
