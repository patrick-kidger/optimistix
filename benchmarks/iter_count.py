"""Compare iteration counts across solvers on the standard test problems."""

import sys


sys.path.insert(0, "/home/user/optimistix")

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import optimistix as optx


jax.config.update("jax_enable_x64", True)

rtol = atol = 1e-4

# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def bowl(tree, args):
    flat, _ = jfu.ravel_pytree(tree)
    return flat @ args @ flat


def matyas(tree, args):
    x, y = tree
    a, b = args
    return a * (x**2 + y**2) - b * x * y


def beale(tree, args):
    x, y = tree
    a, b, c = args
    return (a - x + x * y) ** 2 + (b - x + x * y**2) ** 2 + (c - x + x * y**3) ** 2


def square_minus_one(x, args):
    return jnp.sum(jnp.square(x)) - 1.0


def globally_convex(y, scale):
    return jnp.sum(jnp.cosh(scale * y) - 1)


def _himmelblau(tree, args):
    x, y = tree
    a, b = args
    return (x**2 + y - a) ** 2 + (x + y**2 - b) ** 2


key = jr.PRNGKey(17)
bowl_init = ({"a": 0.05 * jnp.ones((2, 3, 3))}, (0.05 * jnp.ones(2)))
flat_bowl, _ = jfu.ravel_pytree(bowl_init)
matrix = jr.normal(key, (flat_bowl.size, flat_bowl.size))
bowl_args = matrix.T @ matrix

minimisation_problems = [
    ("bowl", bowl, bowl_init, bowl_args, True),
    (
        "matyas",
        matyas,
        [jnp.array(6.0), jnp.array(6.0)],
        (jnp.array(0.26), jnp.array(0.48)),
        True,
    ),
    (
        "beale",
        beale,
        [jnp.array(2.0), jnp.array(0.0)],
        (jnp.array(1.5), jnp.array(2.25), jnp.array(2.625)),
        False,
    ),
    (
        "himmelblau",
        _himmelblau,
        [jnp.array(2.0), jnp.array(2.5)],
        (jnp.array(11.0), jnp.array(7.0)),
        False,
    ),
    ("sq_minus_one", square_minus_one, jnp.array(1.0), None, True),
    ("glob_convex", globally_convex, jnp.array([0.4, -0.3, 0.2]), jnp.ones(3), True),
    (
        "glob_convex_far",
        globally_convex,
        jnp.array([10.0, -8.0, 6.0]),
        jnp.ones(3),
        True,
    ),
]

# ---------------------------------------------------------------------------
# Least-squares problems
# ---------------------------------------------------------------------------


def diagonal_quadratic_bowl(tree, args):
    return jtu.tree_map(lambda x, w: jnp.square(x) * (0.1 + w), tree, args)


diagonal_bowl_init = ({"a": 0.05 * jnp.ones((2, 3, 3))}, (0.05 * jnp.ones(2)))
leaves, treedef = jtu.tree_flatten(diagonal_bowl_init)
diagonal_bowl_args = treedef.unflatten(
    [jr.normal(key, leaf.shape, leaf.dtype) ** 2 for leaf in leaves]
)


def rosenbrock(tree, args):
    x, y = tree
    scale = args
    return jnp.array([10 * (y - x**2), scale * (1 - x)])


rosenbrock_init1 = [jnp.array(0.5), jnp.array(0.4)]
rosenbrock_init2 = [jnp.array(-0.5), jnp.array(0.1)]

least_squares_problems = [
    ("diag_bowl", diagonal_quadratic_bowl, diagonal_bowl_init, diagonal_bowl_args),
    ("rosenbrock_a", rosenbrock, rosenbrock_init1, jnp.array(1.0)),
    ("rosenbrock_b", rosenbrock, rosenbrock_init2, jnp.array(1.0)),
]

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

minimise_solvers = [
    ("LineSearchNewton(Cholesky)", optx.LineSearchNewton(rtol, atol)),
    (
        "LineSearchNewton(CG)",
        optx.LineSearchNewton(rtol, atol, linear_solver=lx.CG(rtol=1e-6, atol=0.0)),
    ),
    (
        "LineSearchNewton(TruncCG)",
        optx.LineSearchNewton(
            rtol, atol, linear_solver=optx.TruncatedCG(rtol=0.5, atol=0.0)
        ),
    ),
    ("TrustNewton(Cholesky)", optx.TrustNewton(rtol, atol)),
    (
        "TrustNewton(CG)",
        optx.TrustNewton(rtol, atol, linear_solver=lx.CG(rtol=1e-6, atol=0.0)),
    ),
    ("TrustNewton(Steihaug)", optx.TrustNewton(rtol, atol, use_steihaug=True)),
    ("BFGS", optx.BFGS(rtol, atol, use_inverse=False)),
    ("LBFGS", optx.LBFGS(rtol, atol, use_inverse=False)),
    ("GradientDescent", optx.GradientDescent(1.5e-2, rtol, atol)),
]

least_squares_solvers = [
    ("LineSearchNewton(Cholesky)", optx.LineSearchNewton(rtol, atol)),
    ("TrustNewton(Cholesky)", optx.TrustNewton(rtol, atol)),
    ("TrustNewton(Steihaug)", optx.TrustNewton(rtol, atol, use_steihaug=True)),
    (
        "GaussNewton",
        optx.GaussNewton(
            rtol, atol, linear_solver=lx.AutoLinearSolver(well_posed=False)
        ),
    ),
    ("LevenbergMarquardt", optx.LevenbergMarquardt(rtol, atol)),
    ("IndirectLevenbergMarquardt", optx.IndirectLevenbergMarquardt(rtol, atol)),
    ("BFGS", optx.BFGS(rtol, atol, use_inverse=False)),
    ("LBFGS", optx.LBFGS(rtol, atol, use_inverse=False)),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

SPD_FNS = {bowl, matyas, square_minus_one, globally_convex}


def run_minimise(solver, fn, init, args, is_spd):
    tags = frozenset({lx.positive_semidefinite_tag}) if is_spd else frozenset()
    max_steps = 100_000 if isinstance(solver, optx.GradientDescent) else 10_000
    sol = optx.minimise(
        fn, solver, init, args=args, max_steps=max_steps, throw=False, tags=tags
    )
    return int(sol.stats["num_steps"]), sol.result == optx.RESULTS.successful


def run_least_squares(solver, fn, init, args):
    sol = optx.least_squares(fn, solver, init, args=args, max_steps=10_000, throw=False)
    return int(sol.stats["num_steps"]), sol.result == optx.RESULTS.successful


def fmt(n, ok):
    mark = "" if ok else "!"
    return f"{n}{mark}"


def print_table(title, solvers, problems, runner):
    solver_names = [s for s, _ in solvers]
    prob_names = [p[0] for p in problems]

    col_w = max(max(len(s) for s in solver_names), 6) + 2
    prob_w = [max(len(p), 6) + 2 for p in prob_names]

    header = f"{'Solver':<{col_w}}" + "".join(
        f"{p:>{w}}" for p, w in zip(prob_names, prob_w)
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(title)
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for sname, solver in solvers:
        row = f"{sname:<{col_w}}"
        for prob, prob_w_i in zip(problems, prob_w):
            prob_args = prob[1:]  # fn, init, args[, is_spd]
            try:
                n, ok = runner(solver, *prob_args)
                row += f"{fmt(n, ok):>{prob_w_i}}"
            except Exception:
                row += f"{'ERR':>{prob_w_i}}"
        print(row)

    print("\n  ! = did not converge (result != successful)")


print_table(
    "MINIMISE  — iterations (! = did not converge)",
    minimise_solvers,
    minimisation_problems,
    lambda solver, fn, init, args, is_spd: run_minimise(solver, fn, init, args, is_spd),
)

print_table(
    "LEAST SQUARES  — iterations (! = did not converge)",
    least_squares_solvers,
    least_squares_problems,
    lambda solver, fn, init, args: run_least_squares(solver, fn, init, args),
)
