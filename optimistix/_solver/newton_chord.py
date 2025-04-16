import warnings
from collections.abc import Callable
from typing import Any, Generic, Optional, TYPE_CHECKING, Union

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Constraint, EqualityOut, Fn, InequalityOut, Out, Y
from .._misc import cauchy_termination, max_norm, tree_clip, tree_full_like
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


# Note: With the introduction of boundary maps, the root finders *could* now
# use these too. However, we still use `clip` below since:
# - The other boundary maps would not necessarily make sense in the context of root
# finders, and if we support one we support them all.
# - This is consistent with the previously behaviour, where bounds were passed as
# options to the solver, and the iterates were clipped to this value.


def _small(diffsize: Scalar) -> Bool[Array, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[Array, " "]:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: float) -> Bool[Array, " "]:
    return (factor > 0) & (factor < tol)


class _NewtonChordState(eqx.Module, Generic[Y], strict=True):
    f: PyTree[Array]
    linear_state: Optional[tuple[lx.AbstractLinearOperator, PyTree]]
    diff: Y
    diffsize: Scalar
    diffsize_prev: Scalar
    result: RESULTS
    step: Scalar


class _NoAux(eqx.Module, strict=True):
    fn: Callable

    def __call__(self, y, args):
        out, aux = self.fn(y, args)
        del aux
        return out


class _AbstractNewtonChord(
    AbstractRootFinder[Y, Out, Aux, _NewtonChordState[Y]], strict=True
):
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    kappa: float = 1e-2
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    cauchy_termination: bool = True

    _is_newton: AbstractClassVar[bool]

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _NewtonChordState[Y]:
        del aux_struct

        clip = options.get("clip", False)
        if clip and bounds is None:
            raise ValueError("Option `'clip'=True` requires bounds to be passed.")
        if clip is False and bounds is not None:  # Catch for smoother upgrades
            warnings.warn(
                "Bounds were passed without clipping enabled. To enable clipping, pass "
                "options=dict(clip=True) to the solver."
            )

        if self._is_newton:
            linear_state = None
        else:
            jac = lx.JacobianLinearOperator(_NoAux(fn), y, args, tags=tags)
            jac = lx.linearise(jac)
            init_later_state = self.linear_solver.init(jac, options={})
            init_later_state = lax.stop_gradient(init_later_state)
            linear_state = (jac, init_later_state)
        if self.cauchy_termination:
            f_val = tree_full_like(f_struct, jnp.inf)
        else:
            f_val = None
        return _NewtonChordState(
            f=f_val,
            linear_state=linear_state,
            diff=tree_full_like(y, jnp.inf),
            diffsize=jnp.array(jnp.inf),
            diffsize_prev=jnp.array(1.0),
            result=RESULTS.successful,
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _NewtonChordState[Y],
        tags: frozenset[object],
    ) -> tuple[Y, _NewtonChordState[Y], Aux]:
        del constraint

        clip = options.get("clip", False)  # Default
        if bounds is not None:
            lower, upper = bounds
        elif options.get("lower") is not None or options.get("upper") is not None:
            lower = options.get("lower", None)
            upper = options.get("upper", None)
            if lower is None and upper is not None:
                lower = tree_full_like(upper, -jnp.inf)
            if upper is None and lower is not None:
                upper = tree_full_like(lower, jnp.inf)
            warnings.warn(
                "Passing bounds through the 'options' dictionary is deprecated. "
                "Pass a pair (lower, upper) through the bounds argument instead and "
                "enable clipping, e.g. with "
                "`optx.root_find(..., options=dict(clip=True), bounds=(lower, upper))`."
            )
            clip = True
        else:
            lower = upper = None
        del options

        if self._is_newton:
            fx, lin_fn, aux = jax.linearize(lambda _y: fn(_y, args), y, has_aux=True)
            jac = lx.FunctionLinearOperator(
                lin_fn, jax.eval_shape(lambda: y), tags=tags
            )
            sol = lx.linear_solve(jac, fx, self.linear_solver, throw=False)
        else:
            fx, aux = fn(y, args)
            jac, linear_state = state.linear_state  # pyright: ignore
            linear_state = lax.stop_gradient(linear_state)
            sol = lx.linear_solve(
                jac, fx, self.linear_solver, state=linear_state, throw=False
            )

        diff = sol.value
        new_y = (y**ω - diff**ω).ω
        if clip:
            new_y = tree_clip(new_y, lower, upper)
            diff = (y**ω - new_y**ω).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        with jax.numpy_dtype_promotion("standard"):
            diffsize = self.norm((diff**ω / scale**ω).ω)

        if self.cauchy_termination:
            f_val = fx
        else:
            f_val = None

        new_state = _NewtonChordState(
            f=f_val,
            linear_state=state.linear_state,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=RESULTS.promote(sol.result),
            step=state.step + 1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _NewtonChordState[Y],
        tags: frozenset[object],
    ):
        del fn, args, options
        if self.cauchy_termination:
            # Compare `f_val` against 0, not against some `f_prev`. This is because
            # we're doing a root-find and know that we're aiming to get close to zero.
            # Note that this does mean that the `rtol` is ignored in f-space, and only
            # `atol` matters.
            terminate = cauchy_termination(
                self.rtol,
                self.atol,
                self.norm,
                y,
                state.diff,
                jtu.tree_map(jnp.zeros_like, state.f),
                state.f,
            )
            terminate_result = RESULTS.successful
        else:
            # TODO(kidger): perform only one iteration when solving a linear system!
            at_least_two = state.step >= 2
            rate = state.diffsize / state.diffsize_prev
            factor = state.diffsize * rate / (1 - rate)
            small = _small(state.diffsize)
            diverged = _diverged(rate)
            converged = _converged(factor, self.kappa)
            terminate = at_least_two & (small | diverged | converged)
            terminate_result = RESULTS.where(
                jnp.invert(small) & (diverged | jnp.invert(converged)),
                RESULTS.nonlinear_divergence,
                RESULTS.successful,
            )
        linsolve_fail = state.result != RESULTS.successful
        result = RESULTS.where(linsolve_fail, state.result, terminate_result)
        terminate = linsolve_fail | terminate
        return terminate, result

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _NewtonChordState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class Newton(_AbstractNewtonChord[Y, Out, Aux], strict=True):
    """Newton's method for root finding. Also sometimes known as Newton--Raphson.

    Unlike the SciPy implementation of Newton's method, the Optimistix version also
    works for vector-valued (or PyTree-valued) `y`.

    This solver accepts the following `options`:

    - `clip`: Whether to clip the iterates of `y` to the hypercube defined by the
        `lower` and `upper` bounds. If `True`, then bounds must be passed as well.
        Defaults to False.
    """

    _is_newton = True


class Chord(_AbstractNewtonChord[Y, Out, Aux], strict=True):
    """The Chord method of root finding.

    This is equivalent to the Newton method, except that the Jacobian is computed only
    once at the initial point `y0`, and then reused throughout the computation. This is
    a useful way to cheapen the solve, if `y0` is expected to be a good initial guess
    and the target function does not change too rapidly. (For example this is the
    standard technique used in implicit Runge--Kutta methods, when solving differential
    equations.)

    This solver optionally accepts the following `options`:

    - `clip`: Whether to clip the iterates of `y` to the hypercube defined by the
        `lower` and `upper` bounds. If `True`, then bounds must be passed as well.
        Defaults to False.
    """

    _is_newton = False


_init_doc = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `kappa`: A tolerance for early convergence check when `cauchy_termination=False`.
- `linear_solver`: The linear solver used to compute the Newton step.
- `cauchy_termination`: When `True`, use the Cauchy termination condition, that
    two adjacent iterates should have a small difference between them. This is usually
    the standard choice when solving general root finding problems. When `False`, use
    a procedure which attempts to detect slow convergence, and quickly fail the solve
    if so. This is useful when iteratively performing the root-find, refining the
    target problem for those which fail. This comes up when solving differential
    equations with adaptive step sizing and implicit solvers. The exact procedure is as
    described in Section IV.8 of Hairer & Wanner, "Solving Ordinary Differential
    Equations II".
"""

Newton.__init__.__doc__ = _init_doc
Chord.__init__.__doc__ = _init_doc
