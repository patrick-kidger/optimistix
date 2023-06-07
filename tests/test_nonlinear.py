import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax as lx
import pytest
from equinox.internal import ω

import optimistix as optx

from .helpers import (
    bisection_fn_init_options_args,
    finite_difference_jvp,
    fixed_point_fn_init_args,
    least_squares_fn_minima_init_args,
    minimisation_fn_minima_init_args,
    shaped_allclose,
    simple_nn,
    trigonometric,
    trivial,
    variably_dimensioned,
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
# - GradOnly (NonlinearCG, GradientDescent)
# - Running Min
# - Optax
#
# LINE SEARCHES:
# - BacktrackingArmijo
# - ClassicalTrustRegion
# - LearningRate
#
# DESCENTS:
# - DirectIterativeDual(GN=...)
# - IndirectIterativeDual(GN=...)
# - Dogleg(GN=...)
# - Newton(GN=...)
# - Gradient (Not appropriate for GN!)
# - NonlinearCG (Not appropriate for GN!)
#
atol = rtol = 1e-8
_lsqr_only = (
    optx.LevenbergMarquardt(rtol, atol),  # DirectIterativeDual(GN=True)
    optx.IndirectLevenbergMarquardt(rtol, atol),  # IndirectIterativeDual(GN=True)
    optx.GaussNewton(rtol, atol),  # Newton(GN=True), LearningRate
    optx.GaussNewton(
        rtol,
        atol,
        line_search=optx.ClassicalTrustRegion(),
        descent=optx.Dogleg(gauss_newton=True),  # Dogleg(GN=True)
    ),
    optx.GaussNewton(
        rtol,
        atol,
        line_search=optx.ClassicalTrustRegion(),
        descent=optx.NormalisedNewton(gauss_newton=True),  # Newton(GN=True)
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
    optx.BFGS(
        rtol,
        atol,
        line_search=optx.ClassicalTrustRegion(),
        descent=optx.IndirectIterativeDual(gauss_newton=False, lambda_0=1.0),
    ),  # IndirectIterativeDual(GN=False)
    optx.BFGS(
        rtol,
        atol,
        line_search=optx.ClassicalTrustRegion(),
        descent=optx.Dogleg(gauss_newton=False),  # Dogleg(GN=False)
    ),
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
    optx.NonlinearCG(
        rtol,
        atol,
        line_search=optx.BacktrackingArmijo(
            gauss_newton=False, backtrack_slope=0.1, decrease_factor=0.5
        ),
    ),  # NonlinearCG
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
# - Bisection (is initialized elsewhere)
#

atol = rtol = 1e-6
_root_finders = (
    optx.Newton(rtol, atol),
    optx.Chord(rtol, atol),
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
# This is just to make sure that aux is handled correctly by the solvers.
#

smoke_aux = (jnp.ones((2, 3)), {"smoke_aux": jnp.ones(2)})


@pytest.mark.parametrize("solver", _lsqr_minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", least_squares_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_least_squares(solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-4
    ignore_solver = (
        isinstance(solver, optx.IndirectLevenbergMarquardt)
        or (
            isinstance(solver, optx.BFGS)
            & isinstance(solver.descent, optx.IndirectIterativeDual)
        )
        or (
            isinstance(solver, optx.GradOnly)
            & isinstance(solver.descent, optx.NormalisedGradient)
        )
    )
    ignore_problem = (_fn == trigonometric) or (_fn == variably_dimensioned)
    if ignore_solver or ignore_problem:
        pytest.skip()

    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    optx_argmin = optx.least_squares(
        fn,
        solver,
        init,
        has_aux=has_aux,
        args=args,
        max_steps=10_000,
        throw=False,
    ).value
    out = fn(optx_argmin, args)
    if has_aux:
        lst_sqr, _ = out
    else:
        lst_sqr = out
    optx_min = jtu.tree_reduce(
        lambda x, y: x + y, jtu.tree_map(lambda x: jnp.sum(x**2), lst_sqr)
    )
    assert shaped_allclose(optx_min, minimum, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", minimisation_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_minimise(solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-4

    if isinstance(solver, optx.BFGS) & isinstance(
        solver.descent, optx.IndirectIterativeDual
    ):
        pytest.skip()

    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn
    optx_argmin = optx.minimise(
        fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
    ).value
    out = fn(optx_argmin, args)
    if has_aux:
        optx_min, _ = out
    else:
        optx_min = out
    assert shaped_allclose(optx_min, minimum, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _fp_solvers)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_fixed_point(solver, _fn, init, args, has_aux):
    atol = rtol = 1e-4
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn
    optx_fp = optx.fixed_point(
        fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
    ).value
    out = fn(optx_fp, args)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    assert shaped_allclose(optx_fp, f_val, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_root_find(solver, _fn, init, args, has_aux):
    atol = rtol = 1e-4

    def root_find_problem(y, args):
        f_val = _fn(y, args)
        return (f_val**ω - y**ω).ω

    if has_aux:
        fn = lambda x, args: (root_find_problem(x, args), smoke_aux)
    else:
        fn = root_find_problem
    optx_fp = optx.root_find(
        fn, solver, init, has_aux=has_aux, args=args, max_steps=10_000, throw=False
    ).value
    out = fn(optx_fp, args)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    zeros = jtu.tree_map(jnp.zeros_like, f_val)
    assert shaped_allclose(f_val, zeros, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "_fn, init, bisection_options, args", bisection_fn_init_options_args
)
@pytest.mark.parametrize("has_aux", (True, False))
def test_bisection(_fn, init, bisection_options, args, has_aux):
    solver = optx.Bisection(rtol=1e-6, atol=1e-6)
    atol = rtol = 1e-4

    def root_find_problem(y, args):
        f_val = _fn(y, args)
        return (f_val**ω - y**ω).ω

    if _fn == trivial:
        bisection_problem = _fn
    else:
        bisection_problem = root_find_problem
    if has_aux:
        fn = lambda x, args: (bisection_problem(x, args), smoke_aux)
    else:
        fn = bisection_problem

    optx_fp = optx.root_find(
        fn,
        solver,
        init,
        has_aux=has_aux,
        args=args,
        options=bisection_options,
        max_steps=10_000,
        throw=False,
    ).value
    out = fn(optx_fp, args)
    if has_aux:
        f_val, _ = out
    else:
        f_val = out
    zeros = jtu.tree_map(jnp.zeros_like, f_val)
    assert shaped_allclose(f_val, zeros, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _lsqr_minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", least_squares_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_least_squares_jvp(getkey, solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-2
    ignore_solver = (
        isinstance(solver, optx.IndirectLevenbergMarquardt)
        or (
            isinstance(solver, optx.BFGS)
            & isinstance(solver.descent, optx.IndirectIterativeDual)
        )
        or (
            isinstance(solver, optx.GradOnly)
            & isinstance(solver.descent, optx.NormalisedGradient)
        )
    )
    ignore_problem = (_fn == trigonometric) or (_fn == variably_dimensioned)
    if ignore_solver or ignore_problem:
        pytest.skip()
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def least_squares(x, dynamic_args):
        args = eqx.combine(dynamic_args, static_args)

        if _fn == simple_nn:
            adjoint = optx.ImplicitAdjoint(lx.AutoLinearSolver(well_posed=False))
        else:
            adjoint = optx.ImplicitAdjoint()

        return optx.least_squares(
            fn,
            solver,
            x,
            has_aux=has_aux,
            args=args,
            adjoint=adjoint,
            max_steps=10_000,
            throw=False,
        ).value

    optx_argmin = least_squares(init, args)
    expected_out, t_expected_out = finite_difference_jvp(
        least_squares, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    out, t_out = eqx.filter_jvp(
        least_squares, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _minimisers)
@pytest.mark.parametrize("_fn, minimum, init, args", minimisation_fn_minima_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_minimise_jvp(getkey, solver, _fn, minimum, init, args, has_aux):
    atol = rtol = 1e-4

    if isinstance(solver, optx.BFGS) & isinstance(
        solver.descent, optx.IndirectIterativeDual
    ):
        pytest.skip()

    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def minimise(x, dynamic_args):
        args = eqx.combine(dynamic_args, static_args)
        return optx.minimise(
            fn, solver, x, has_aux=has_aux, args=args, max_steps=10_000, throw=False
        ).value

    optx_argmin = minimise(init, dynamic_args)
    expected_out, t_expected_out = finite_difference_jvp(
        minimise, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    out, t_out = eqx.filter_jvp(
        minimise, (optx_argmin, dynamic_args), (t_init, t_dynamic_args)
    )
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _fp_solvers)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_fixed_point_jvp(getkey, solver, _fn, init, args, has_aux):
    atol = rtol = 1e-4
    if has_aux:
        fn = lambda x, args: (_fn(x, args), smoke_aux)
    else:
        fn = _fn

    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    def fixed_point(x, dynamic_args):
        args = eqx.combine(dynamic_args, static_args)
        return optx.fixed_point(
            fn, solver, x, has_aux=has_aux, args=args, max_steps=10_000
        ).value

    optx_fp = fixed_point(init, dynamic_args)
    expected_out, t_expected_out = finite_difference_jvp(
        fixed_point, (optx_fp, dynamic_args), (t_init, t_dynamic_args)
    )
    out, t_out = eqx.filter_jvp(
        fixed_point, (optx_fp, dynamic_args), (t_init, t_dynamic_args)
    )
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("solver", _root_finders)
@pytest.mark.parametrize("_fn, init, args", fixed_point_fn_init_args)
@pytest.mark.parametrize("has_aux", (True, False))
def test_root_find_jvp(getkey, solver, _fn, init, args, has_aux):
    atol = rtol = 1e-4

    def root_find_problem(y, args):
        f_val = _fn(y, args)
        return (f_val**ω - y**ω).ω

    if has_aux:
        fn = lambda x, args: (root_find_problem(x, args), smoke_aux)
    else:
        fn = root_find_problem
    dynamic_args, static_args = eqx.partition(args, eqx.is_array)
    t_init = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), init)
    t_dynamic_args = jtu.tree_map(lambda x: jr.normal(getkey(), x.shape), dynamic_args)

    # Chord struggles to hit the very min, it get's close enough to pass and then
    # continues to decrease at a slow rate without signaling convergence for a
    # while.
    def root_find(x, dynamic_args):
        args = eqx.combine(dynamic_args, static_args)
        return optx.root_find(
            fn, solver, x, has_aux=has_aux, args=args, max_steps=10_000, throw=False
        ).value

    optx_root = root_find(init, dynamic_args)
    expected_out, t_expected_out = finite_difference_jvp(
        root_find, (optx_root, dynamic_args), (t_init, t_dynamic_args)
    )
    out, t_out = eqx.filter_jvp(
        root_find, (optx_root, dynamic_args), (t_init, t_dynamic_args)
    )
    assert shaped_allclose(out, expected_out, atol=atol, rtol=rtol)
