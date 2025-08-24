# copy the imports at https://github.com/patrick-kidger/optimistix/blob/c1dad7e75fc35bd5a4977ac3a872991e51e83d2c/optimistix/_solver/fixed_point.py
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._fixed_point import AbstractFixedPointSolver
from .._misc import max_norm
from .._solution import RESULTS
from .._solver.fixed_point import _FixedPointState


# although Varadhan and Roland (2008) provide 3 different schemes
# they assumed a 1D-vector. Only their recommenedd scheme (scheme 3) can be generalised
# to arbitrary dimensions through the Frobenius norm.
# We leave the alternate schemes here as dead code for reference.
# def s1(r: Y, v: Y):
#     r"""Scheme 1 of SQUAREM algorithm steplength

#     Equation (7) of Varadhan and Roland (2008)
#     for the computation of steplength $alpha = \frac{r^Tv}{v^Tv}$.
#     """
#     return (ω(r).call(jnp.transpose) @ v**ω).ω / (ω(v).call(jnp.transpose) @ v**ω).ω


# def s2(r: Y, v: Y):
#     r"""Scheme 2 of SQUAREM algorithm steplength

#     Equation (8) of Varadhan and Roland (2008)
#     for the computation of steplength $\alpha = \frac{r^Tr}{r^Tv}$.
#     """
#     return (ω(r).call(jnp.transpose) @ r**ω).ω / (ω(r).call(jnp.transpose) @ v**ω).ω


def s3(r: Y, v: Y):
    r"""Scheme 3 of SQUAREM algorithm steplength

    Equation (9) of Varadhan and Roland (2008)
    for the computation of $\alpha = \frac{\left\|r^Tr\right\|{\left\|r^Tv\right\|$.
    We use the Frobenius norm to accommodate arrays with many axes.
    """
    # compute the L2 norms of r and v
    l2_r = ω(r).call(lambda x: jnp.linalg.norm(x, ord=None))
    l2_v = ω(v).call(lambda x: jnp.linalg.norm(x, ord=None))
    return (-1.0 * l2_r / l2_v).ω


# optimistix compliant implementation of SQUAREM accelerated fixed point
class SquarEM(AbstractFixedPointSolver[Y, Aux, _FixedPointState]):
    r"""SQUAREM acceleration for fixed point equations per Varadhan and Roland (2008).
    As we are using it purely for fixed point acceleration, we do not implement
    the option of passing in a likelihood function (useful when SQUAREM is used for EM
    estimation).

    ??? cite "References"
        ```bibtex
        @article{varadhan2008simple,
            title={Simple and globally convergent methods for
                   accelerating the convergence of any EM algorithm},
            author={Varadhan, Ravi and Roland, Christophe},
            journal={Scandinavian Journal of Statistics},
            volume={35},
            number={2},
            pages={335--353},
            year={2008},
            publisher={Wiley Online Library}
        }
        ```
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    scheme: Callable[[PyTree, PyTree], Scalar] = s3

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Scalar] = max_norm,
        method: Callable[[Y, Y], Scalar] = s3,
    ):
        """**Arguments:**

        - `rtol`: Relative tolerance for terminating solve.
        - `atol`: Absolute tolerance for terminating solve.
        - `norm`: The norm used to determine the difference between two iterates in the
            convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
            includes three built-in norms: [`optimistix.max_norm`][],
            [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
        - `method`: The function which computes the steplength `alpha`.
            Only Scheme 3 using Frobenius norms is implemented as the
            other schemes do not generalise to parameters with many
            axes.
        """
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.scheme = method

    def init(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _FixedPointState:
        del fn, y, args, options, f_struct, aux_struct
        return _FixedPointState(jnp.array(jnp.inf))

    def step(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Y, _FixedPointState, Aux]:
        # step twice
        y1, _ = fn(y, args)
        y2, _ = fn(y1, args)
        # compute r, v in the paper
        r = (y1**ω - y**ω).ω
        v = (y2**ω - y1**ω - r**ω).ω
        # compute the alpha based on scheme chosen at initialisation
        with jax.numpy_dtype_promotion("standard"):
            alpha = self.scheme(r, v)
            # per Varadhan and Roland (2008), set the stepsize to be at most -1
            alpha = ω(alpha).call(lambda x: jnp.minimum(x, -1.0))
            y_prime = (y**ω - 2.0 * alpha * r**ω + alpha**2 * v**ω).ω
        # last stabilisation step
        new_y, aux = fn(y_prime, args)
        error = (y**ω - new_y**ω).ω
        with jax.numpy_dtype_promotion("standard"):
            scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
            new_state = _FixedPointState(self.norm((error**ω / scale**ω).ω))
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.relative_error < 1, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


SquarEM.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `scheme`: Scheme used to compute the steplength $alpha$.
    Defaults to s3 (the recommended scheme).
"""
