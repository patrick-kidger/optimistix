from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import optimistix as optx
import pytest
from jaxtyping import PyTree, Scalar

from .helpers import (
    BFGSDampedNewton,
    least_squares_optimisers,
    minimisers,
    tree_allclose,
)


def test_cauchy_convergence_check():
    convergence = optx.CauchyConvergence(rtol=1e-3, atol=1e-6)
    y = jnp.array([1.0, 2.0])
    f_info = optx.FunctionInfo.Eval(jnp.array(3.0))
    small = jnp.array([1e-7, 1e-7])
    large = jnp.array([1e-1, 1e-1])
    assert convergence.check(y, f_info, small, jnp.array(1e-7))
    assert not convergence.check(y, f_info, large, jnp.array(1e-7))
    assert not convergence.check(y, f_info, small, jnp.array(1.0))
    assert convergence.norm is optx.max_norm


def test_cauchy_convergence_positional_arguments():
    convergence = optx.CauchyConvergence(1e-3, 1e-6, optx.two_norm)
    assert convergence.rtol == 1e-3
    assert convergence.atol == 1e-6
    assert convergence.norm is optx.two_norm


class _CustomFunctionInfo(optx.FunctionInfo):
    value: Scalar

    def as_min(self):
        return self.value


_y = jnp.array(1.0)
_f = jnp.array(6.0)
_operator = lx.IdentityLinearOperator(jax.eval_shape(lambda: _y))
_f_infos = (
    optx.FunctionInfo.Eval(_f),
    optx.FunctionInfo.EvalGrad(_f, _y),
    optx.FunctionInfo.EvalGradHessian(_f, _y, _operator),
    optx.FunctionInfo.EvalGradHessianInv(_f, _y, _operator),
    optx.FunctionInfo.Residual(_f),
    optx.FunctionInfo.ResidualJac(_f, _operator),
    _CustomFunctionInfo(_f),
)


# Each of these records `f = 6`, so `f_diff` is compared against `1e-6 + 1e-3 * 6`.
@pytest.mark.parametrize("f_info", _f_infos)
def test_cauchy_convergence_function_info(f_info):
    convergence = optx.CauchyConvergence(rtol=1e-3, atol=1e-6)
    y_diff = jnp.array(1e-7)
    assert convergence.check(_y, f_info, y_diff, jnp.array(1e-3))
    assert not convergence.check(_y, f_info, y_diff, jnp.array(1e-2))


class _AbsoluteConvergence(optx.AbstractConvergence):
    ytol: float
    ftol: float
    norm: Callable[[PyTree], Scalar] = optx.max_norm

    def check(self, y, f_info, y_diff, f_diff):
        y_converged = self.norm(y_diff) < self.ytol
        f_converged = self.norm(f_diff) < self.ftol
        return y_converged & f_converged


class _MaxNormConvergence(optx.AbstractConvergence):
    norm: Callable[[PyTree], Scalar]

    def check(self, y, f_info, y_diff, f_diff):
        y_converged = optx.max_norm(y_diff) < 1e-8
        f_converged = optx.max_norm(f_diff) < 1e-8
        return y_converged & f_converged


@pytest.mark.parametrize("solver", minimisers)
def test_minimiser_convergence_attribute(solver):
    assert isinstance(solver.convergence, optx.AbstractConvergence)


@pytest.mark.parametrize("solver", least_squares_optimisers)
def test_least_squares_convergence_attribute(solver):
    assert isinstance(solver.convergence, optx.AbstractConvergence)


@pytest.mark.parametrize(
    "solver",
    (
        optx.BestSoFarMinimiser(optx.BFGS(rtol=1e-3, atol=1e-6, norm=optx.two_norm)),
        optx.BestSoFarLeastSquares(
            optx.LevenbergMarquardt(rtol=1e-3, atol=1e-6, norm=optx.two_norm)
        ),
    ),
)
def test_best_so_far_convergence_attribute(solver):
    assert isinstance(solver.convergence, optx.CauchyConvergence)
    assert solver.convergence.rtol == 1e-3
    assert solver.convergence.atol == 1e-6
    assert solver.convergence.norm is optx.two_norm


def test_convergence_attribute_values():
    solver = optx.BFGS(rtol=1e-3, atol=1e-6, norm=optx.two_norm)
    assert isinstance(solver.convergence, optx.CauchyConvergence)
    assert solver.convergence.rtol == 1e-3
    assert solver.convergence.atol == 1e-6
    assert solver.convergence.norm is optx.two_norm


def test_golden_search_convergence_attribute():
    solver = optx.GoldenSearch(rtol=1e-3, atol=1e-6)
    assert isinstance(solver.convergence, optx.CauchyConvergence)
    assert solver.convergence.rtol == 1e-3
    assert solver.convergence.atol == 1e-6
    assert solver.convergence.norm is jnp.abs


def test_convergence_attribute_custom():
    convergence = _AbsoluteConvergence(ytol=1e-6, ftol=1e-6)
    solver = BFGSDampedNewton(convergence)
    assert solver.convergence is convergence
    assert solver.convergence.norm is optx.max_norm


@pytest.mark.parametrize(
    "solver",
    (
        optx.Newton(rtol=1e-3, atol=1e-6, norm=optx.two_norm),
        optx.BestSoFarRootFinder(optx.Newton(rtol=1e-3, atol=1e-6, norm=optx.two_norm)),
        optx.FixedPointIteration(rtol=1e-3, atol=1e-6, norm=optx.two_norm),
        optx.BestSoFarFixedPoint(
            optx.FixedPointIteration(rtol=1e-3, atol=1e-6, norm=optx.two_norm)
        ),
    ),
)
def test_root_finder_and_fixed_point_tolerance_attributes(solver):
    assert solver.rtol == 1e-3
    assert solver.atol == 1e-6
    assert solver.norm is optx.two_norm


def _root_fn(y, _):
    ya, (yb, yc) = y
    return jnp.tanh(ya + 0.1), jnp.tanh(yb - 0.5), jnp.tanh(yc * 2)


def _bad_root_fn(y, _):
    ya, (yb, yc) = y
    return 1.0, jnp.tanh(yb - 0.5), jnp.tanh(yc * 2)


_root_y0 = (jnp.array(0.5), jnp.array([-0.3, 0.7]))
_root_expected = (jnp.array(-0.1), jnp.array([0.5, 0.0]))


def test_root_via_min_custom_convergence():
    solver = BFGSDampedNewton(_AbsoluteConvergence(ytol=1e-8, ftol=1e-8))
    sol = optx.root_find(_root_fn, solver, _root_y0)
    assert tree_allclose(sol.value, _root_expected)


def test_bad_root_via_min_custom_convergence():
    solver = BFGSDampedNewton(_AbsoluteConvergence(ytol=1e-8, ftol=1e-8))
    sol = optx.root_find(_bad_root_fn, solver, _root_y0, throw=False)
    assert sol.result == optx.RESULTS.nonlinear_max_steps_reached


def test_root_via_min_uses_convergence_norm():
    structures = []

    def norm(x):
        structures.append(jtu.tree_structure(x))
        return optx.max_norm(x)

    solver = BFGSDampedNewton(_MaxNormConvergence(norm=norm))
    sol = optx.root_find(_root_fn, solver, _root_y0)
    assert tree_allclose(sol.value, _root_expected)
    residual_structure = jtu.tree_structure(_root_fn(_root_y0, None))
    assert len(structures) > 0
    assert all(structure == residual_structure for structure in structures)
