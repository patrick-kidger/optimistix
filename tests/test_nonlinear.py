import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from equinox.internal import ω

import optimistix as optx

from .helpers import finite_difference_jvp, shaped_allclose
from .test_problems import (
    fixed_point_problem_init,
    least_squares_problem_minima_init,
    minimisation_problem_minima_init,
)


def _norm_squared(tree):
    sumsqr = lambda x: jnp.sum(x**2)
    return jtu.tree_reduce(jnp.add, jtu.tree_map(sumsqr, tree))


#
# NOTE: `GN` is shorthand for `gauss_newton`. We want to be sure we test every
# branch of `GN=True` and `GN=False` for all of these solvers.
#
# UNCONSTRAINED MIN
#
# SOLVERS:
# - Gauss-Newton (LM)
# - BFGS
# - GradOnly (NonlinearCG)
# - Running Min
# - Optax
#
# LINE SEARCHES:
# - BacktrackingArmijo
# - ClassicalTrustRegion
#
# DESCENTS:
# - DirectIterativeDual(GN=...)
# - IndirectIterativeDual(GN=...)
# - Dogleg(GN=...)
# - Newton(GN=...) (NewtonInverse)
# - Gradient
# - NonlinearCG
#
atol = rtol = 1e-8
_lsqr_only = (
    optx.LevenbergMarquardt(rtol, atol),  # DirectIterativeDual(GN=True)
    # these two are having problems most likely caused by the convergence criteria.
    # optx.IndirectLevenbergMarquardt(rtol, atol), # IndirectIterativeDual(GN=True)
    # optx.GaussNewton(
    #     rtol,
    #     atol,
    #     line_search=optx.ClassicalTrustRegion(),
    #     descent=optx.Dogleg(gauss_newton=True) # Dogleg(GN=True)
    # ),
    optx.GaussNewton(
        rtol,
        atol,
        line_search=optx.ClassicalTrustRegion(),
        descent=optx.NormalisedNewton(gauss_newton=True),  # Newton(GN=True)
    ),
    optx.GaussNewton(
        rtol,
        atol,
        line_search=optx.BacktrackingArmijo(
            gauss_newton=True, backtrack_slope=0.1, decrease_factor=0.5
        ),
        descent=optx.NormalisedGradient(),  # Gradient
    ),
    optx.GaussNewton(
        rtol,
        atol,
        line_search=optx.BacktrackingArmijo(
            gauss_newton=True, backtrack_slope=0.1, decrease_factor=0.5
        ),
        descent=optx.NonlinearCGDescent(method=optx.polak_ribiere),  # NonlinearCG
    ),
)
# NOTE: Should we parametrize the line search algo?
# NOTE: Should we parametrize the nonlinearCG method?
atol = rtol = 1e-8
_minimisers = (
    optx.BFGS(
        rtol,
        atol,
        line_search=optx.ClassicalTrustRegion(),
        descent=optx.DirectIterativeDual(
            gauss_newton=False
        ),  # DirectIterativeDual(GN=False)
    ),
    # optx.BFGS(
    #     rtol,
    #     atol,
    #     line_search=optx.ClassicalTrustRegion(),
    #     descent=optx.IndirectIterativeDual(gauss_newton=False, lambda_0=1.),
    # ), # IndirectIterativeDual(GN=False)
    # optx.BFGS(
    #     rtol,
    #     atol,
    #     line_search=optx.ClassicalTrustRegion(),
    #     descent=optx.Dogleg(gauss_newton=False), # Dogleg(GN=False)
    # ),
    optx.BFGS(
        rtol,
        atol,
        line_search=optx.BacktrackingArmijo(
            gauss_newton=False, backtrack_slope=0.1, decrease_factor=0.5
        ),
        descent=optx.NormalisedNewton(gauss_newton=False),  # Newton(GN=False)
    ),
    optx.BFGS(
        rtol,
        atol,
        line_search=optx.BacktrackingArmijo(
            gauss_newton=False, backtrack_slope=0.1, decrease_factor=0.5
        ),
        use_inverse=True,  # NewtonInverse
    ),
    optx.GradOnly(
        rtol,
        atol,
        line_search=optx.BacktrackingArmijo(
            gauss_newton=False, backtrack_slope=0.1, decrease_factor=0.5
        ),
        descent=optx.NormalisedGradient(),  # Gradient
    ),
    # optx.NonlinearCG(
    #     rtol,
    #     atol,
    #     line_search=optx.BacktrackingArmijo(
    #         gauss_newton=False, backtrack_slope=0.1, decrease_factor=0.5
    #     ),
    # ), # NonlinearCG
)

# the minimisers can handle least squares problems, but the least squares
# solvers cannot handle general minimisation problems.
_lsqr_minimisers = _lsqr_only + _minimisers

#
# ROOT FIND
#
# SOLVERS:
# - Newton
# - Chord
# - Bisection
#

atol = rtol = 1e-6
_root_finders = (
    # optx.Bisection(rtol, atol),
    optx.Newton(rtol, atol),
    # optx.Chord(rtol, atol),
)

#
# FIXED POINT
#
# SOLVERS:
# - Fixed point iteration
#

atol = rtol = 1e-6
_fp_solvers = (optx.FixedPointIteration(rtol, atol),)

#
# If `has_aux` in any of these we pass the extra PyTree `smoke_aux` as an aux value.
# This is just to make sure that auxs are handled correctly by the solvers, and
#

smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", _lsqr_minimisers)
@pytest.mark.parametrize("_problem, minimum, init", least_squares_problem_minima_init)
@pytest.mark.parametrize("has_aux", (True, False))
def test_least_squares(solver, _problem, minimum, init, has_aux):
    atol = rtol = 1e-4
    if has_aux:

        def aux_problem(x, args):
            return _problem.fn(x, args), smoke_aux

        problem = optx.LeastSquaresProblem(aux_problem, True, _problem.tags)
    else:
        problem = _problem
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    optx_argmin = optx.least_squares(
        problem,
        solver,
        dynamic_init,
        args=static_init,
        max_steps=10_000,
        throw=False,
    ).value
    out = problem.fn(optx_argmin, static_init)
    if has_aux:
        lst_sqr, _ = out
    else:
        lst_sqr = out
    optx_min = jtu.tree_reduce(
        lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x**2), lst_sqr)
    )
    assert shaped_allclose(optx_min, minimum, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _minimisers)
@pytest.mark.parametrize("_problem, minimum, init", minimisation_problem_minima_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_minimise(solver, _problem, minimum, init, has_aux):
    atol = rtol = 1e-4
    if has_aux:

        def aux_problem(x, args):
            return _problem.fn(x, args), smoke_aux

        problem = optx.MinimiseProblem(aux_problem, True, _problem.tags)
    else:
        problem = _problem
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    optx_argmin = optx.minimise(
        problem, solver, dynamic_init, args=static_init, max_steps=10_000, throw=False
    ).value
    out = problem.fn(optx_argmin, static_init)
    if has_aux:
        optx_min, _ = out
    else:
        optx_min = out
    assert shaped_allclose(optx_min, minimum, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _fp_solvers)
@pytest.mark.parametrize("_problem, init", fixed_point_problem_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_fixed_point(solver, _problem, init, has_aux):
    atol = rtol = 1e-4
    if has_aux:

        def aux_problem(x, args):
            return _problem.fn(x, args), smoke_aux

        problem = optx.FixedPointProblem(aux_problem, True, _problem.tags)
    else:
        problem = _problem
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    optx_fp = optx.fixed_point(
        problem, solver, dynamic_init, args=static_init, max_steps=10_000, throw=False
    ).value
    out = problem.fn(optx_fp, static_init)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    assert shaped_allclose(optx_fp, f_val, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_problem, init", fixed_point_problem_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_root_find(solver, _problem, init, has_aux):
    atol = rtol = 1e-4

    def root_find_problem(y, args):
        f_val = _problem.fn(y, args)
        return (f_val**ω - y**ω).ω

    def root_find_problem_aux(y, args):
        f_val = _problem.fn(y, args)
        return (f_val**ω - y**ω).ω, smoke_aux

    if has_aux:
        problem = optx.RootFindProblem(root_find_problem_aux, has_aux=True)
    else:
        problem = optx.RootFindProblem(root_find_problem, has_aux=False)
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    optx_fp = optx.root_find(
        problem, solver, dynamic_init, args=static_init, max_steps=10_000, throw=False
    ).value
    out = _problem.fn(optx_fp, static_init)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    assert shaped_allclose(optx_fp, f_val, atol=atol, rtol=rtol)


@pytest.mark.parametrize("_problem, init", fixed_point_problem_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_bisection(_problem, init, has_aux):
    ...


@pytest.mark.parametrize("solver", _lsqr_minimisers)
@pytest.mark.parametrize("_problem, minimum, init", least_squares_problem_minima_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_least_squares_jvp(getkey, solver, _problem, minimum, init, has_aux):
    atol = rtol = 1e-4
    if has_aux:

        def aux_problem(x, args):
            return _problem.fn(x, args), smoke_aux

        problem = optx.MinimiseProblem(aux_problem, True, _problem.tags)
    else:
        problem = _problem
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    t_dynamic_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_init)

    def least_squares(x):
        return optx.least_squares(
            problem, solver, x, args=static_init, max_steps=10_000
        ).value

    optx_argmin = least_squares(dynamic_init)
    expected_out, t_expected_out = finite_difference_jvp(
        least_squares, (optx_argmin,), (t_dynamic_init,)
    )
    out, t_out = eqx.filter_jvp(least_squares, (optx_argmin,), (t_dynamic_init,))
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)
    assert shaped_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _minimisers)
@pytest.mark.parametrize("_problem, minimum, init", minimisation_problem_minima_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_minimise_jvp(getkey, solver, _problem, minimum, init, has_aux):
    atol = rtol = 1e-4
    if has_aux:

        def aux_problem(x, args):
            return _problem.fn(x, args), smoke_aux

        problem = optx.MinimiseProblem(aux_problem, True, _problem.tags)
    else:
        problem = _problem
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    t_dynamic_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_init)

    def minimise(x):
        return optx.minimise(
            problem, solver, x, args=static_init, max_steps=10_000
        ).value

    optx_argmin = minimise(dynamic_init)
    expected_out, t_expected_out = finite_difference_jvp(
        minimise, (optx_argmin,), (t_dynamic_init,)
    )
    out, t_out = eqx.filter_jvp(minimise, (optx_argmin,), (t_dynamic_init,))
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)
    assert shaped_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _fp_solvers)
@pytest.mark.parametrize("_problem, init", fixed_point_problem_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_fixed_point_jvp(getkey, solver, _problem, minimum, init, has_aux):
    atol = rtol = 1e-4
    if has_aux:

        def aux_problem(x, args):
            return _problem.fn(x, args), smoke_aux

        problem = optx.MinimiseProblem(aux_problem, True, _problem.tags)
    else:
        problem = _problem
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    t_dynamic_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_init)

    def fixed_point(x):
        return optx.fixed_point(
            problem, solver, x, args=static_init, max_steps=10_000
        ).value

    optx_fp = fixed_point(dynamic_init)
    expected_out, t_expected_out = finite_difference_jvp(
        fixed_point, (optx_fp,), (t_dynamic_init,)
    )
    out, t_out = eqx.filter_jvp(fixed_point, (optx_fp,), (t_dynamic_init,))
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)
    assert shaped_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_problem, init", fixed_point_problem_init)
@pytest.mark.parametrize("has_aux", (False,))
def test_root_find_jvp(getkey, solver, _problem, minimum, init, has_aux):
    atol = rtol = 1e-4

    def root_find_problem(y, args):
        f_val = _problem.fn(y, args)
        return (f_val**ω - y**ω).ω

    def root_find_problem_aux(y, args):
        f_val = _problem.fn(y, args)
        return (f_val**ω - y**ω).ω, smoke_aux

    if has_aux:
        problem = optx.RootFindProblem(root_find_problem_aux, has_aux=True)
    else:
        problem = optx.RootFindProblem(root_find_problem, has_aux=False)
    dynamic_init, static_init = eqx.partition(init, eqx.is_inexact_array)
    t_dynamic_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_init)

    def fixed_point(x):
        return optx.root_find(
            problem, solver, x, args=static_init, max_steps=10_000
        ).value

    optx_fp = fixed_point(dynamic_init)
    expected_out, t_expected_out = finite_difference_jvp(
        fixed_point, (optx_fp,), (t_dynamic_init,)
    )
    out, t_out = eqx.filter_jvp(fixed_point, (optx_fp,), (t_dynamic_init,))
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)
    assert shaped_allclose(t_out, t_expected_out, atol=atol, rtol=rtol)
