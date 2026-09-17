import abc
from collections.abc import Callable
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from ._custom_types import Y
from ._misc import max_norm
from ._search import FunctionInfo


_FnInfo = TypeVar("_FnInfo", contravariant=True, bound=FunctionInfo)


def _resolve_f_value(f_info: FunctionInfo) -> PyTree:
    if isinstance(f_info, FunctionInfo.Residual | FunctionInfo.ResidualJac):
        return f_info.residual
    elif isinstance(
        f_info,
        FunctionInfo.Eval
        | FunctionInfo.EvalGrad
        | FunctionInfo.EvalGradHessian
        | FunctionInfo.EvalGradHessianInv,
    ):
        return f_info.f
    else:
        return f_info.as_min()


class AbstractConvergence(eqx.Module, Generic[Y, _FnInfo]):
    """The abstract base class for convergence criteria. A convergence criterion
    consumes the most recently accepted iterate, the [`optimistix.FunctionInfo`][]
    evaluated there, and how much the iterate and the function value changed over the
    last accepted step, then checks whether the solve has converged.

    Minimisers and least-squares solvers each hold a convergence criterion as their
    `convergence` attribute.

    Which information a criterion receives depends on the solver it is used with. A
    criterion that requires particular information should say so in its type
    parameters: a gradient test would subclass
    `AbstractConvergence[Y, FunctionInfo.EvalGrad]`, and can then only be used with
    solvers that evaluate a gradient.
    """

    norm: AbstractVar[Callable[[PyTree], Scalar]]

    @abc.abstractmethod
    def check(
        self, y: Y, f_info: _FnInfo, y_diff: Y, f_diff: PyTree
    ) -> Bool[Array, ""]:
        """Decides whether the solve has converged.

        **Arguments:**

        - `y`: the value of the most recently accepted iterate.
        - `f_info`: an [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, and potentially the gradient of `f` at `y`, etc.
        - `y_diff`: the difference between `y` and the previously accepted iterate.
        - `f_diff`: the difference between the function value (or the residuals, for a
            least-squares problem) at `y` and at the previously accepted iterate.

        **Returns:**

        A boolean scalar, which is `True` if the solve has converged.
        """


class CauchyConvergence(AbstractConvergence[Y, FunctionInfo]):
    """Converged if there is a small difference in both `y` space and `f` space, as
    determined by `rtol` and `atol`.

    Specifically, this checks that `y_diff < atol + rtol * y` and
    `f_diff < atol + rtol * f`, as measured by `norm`, and reports convergence when
    both of these are true. Here `f` is taken from `f_info`: the residuals for a
    least-squares problem, and the function value otherwise.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm

    def check(
        self, y: Y, f_info: FunctionInfo, y_diff: Y, f_diff: PyTree
    ) -> Bool[Array, ""]:
        f = _resolve_f_value(f_info)
        y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
        f_scale = (self.atol + self.rtol * ω(f).call(jnp.abs)).ω
        y_converged = self.norm((ω(y_diff).call(jnp.abs) / y_scale**ω).ω) < 1
        f_converged = self.norm((ω(f_diff).call(jnp.abs) / f_scale**ω).ω) < 1
        return y_converged & f_converged


CauchyConvergence.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
"""
