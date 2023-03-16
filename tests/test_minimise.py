import jax
import jax.numpy as jnp
import pytest
from equinox.internal import ω

import optimistix as optx

from .helpers import shaped_allclose


def _square(x):
    return ω(x).call(lambda x: x**2).ω


###
# Some standard test functions for nonlinear minimisation. All of these can be found
# on the wikipedia page: https://en.wikipedia.org/wiki/Test_functions_for_optimization
###


def _himmelblau(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = (x**2 + y - 11.0) ** 2
    term2 = (x + y**2 - 7.0) ** 2
    return term1 + term2


def _matyas(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = 0.26 * (x**2 + y**2)
    term2 = 0.48 * x * y
    return term1 - term2


def _eggholder(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = -(y + 47.0) * jnp.sin(jnp.sqrt(jnp.abs(0.5 * x + (y + 47.0))))
    term2 = -x * jnp.sin(jnp.sqrt(jnp.abs(x - (y + 47.0))))
    return term1 + term2


def _beale(z, args):
    (x, y), _ = jax.flatten_util.ravel_pytree(z)
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y**2) ** 2
    term3 = (2.625 - x + x * y**3) ** 2
    return term1 + term2 + term3


###
# Some nonlinear least squares test problems. These are taken from
# J. Moré, B. Garbow, and K. Hillstrom "Testing Unconstrained Optimization Software:"
# https://www.osti.gov/servlets/purl/6650344.
# See also the Fortran package TEST_NLS:
# https://people.math.sc.edu/Burkardt/f_src/test_nls/test_nls.html.
###

_fn_and_init = (
    (
        optx.MinimiseProblem(_himmelblau, False),
        [jnp.array(1.0), jnp.array(1.0)],
        [jnp.array(3.0), jnp.array(2.0)],
    ),
    (
        optx.MinimiseProblem(_matyas, False),
        [6.0 * jnp.array(1.0), 6.0 * jnp.array(1.0)],
        ([jnp.array(0.0), jnp.array(0.0)]),
    ),
    (
        optx.MinimiseProblem(_beale, False),
        [jnp.array(2.0), jnp.array(0.0)],
        [jnp.array(3.0), jnp.array(0.5)],
    ),
)


###
# TODO(Jason): add tests for least_square solvers as well.
###

_minimisers = ((optx.NelderMead, (1e-4, 1e-5)),)


@pytest.mark.parametrize("solver, init_args", _minimisers)
@pytest.mark.parametrize("problem, init, result_exact", _fn_and_init)
def test_minimise(solver, init_args, problem, init, result_exact):
    solver = solver(*init_args)
    solver.init(problem, init, None, {})
    result_optx = optx.minimise(problem, solver, init, max_steps=1024)

    assert shaped_allclose(result_optx.value, result_exact, atol=1e-4, rtol=1e-5)
