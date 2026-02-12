import abc
from collections.abc import Callable
from typing import Generic, TypeVar
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from ._custom_types import Y
from ._misc import max_norm


_F = TypeVar("_F")


class AbstractTermination(eqx.Module, Generic[Y]):
    """TODO"""

    @abc.abstractmethod
    def __call__(self, y: Y, y_diff: Y, f: _F, f_diff: _F) -> Bool[Array, ""]:
        """TODO"""


class CauchyTermination(AbstractTermination[Y]):
    """Terminate if there is a small difference in both `y` space
    and `f` space, as determined by `rtol` and `atol`.

    Specifically, this checks that `y_diff < atol + rtol * y` and
    `f_diff < atol + rtol * f_prev`, terminating if both of these are true.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm

    @override
    def __call__(self, y: Y, y_diff: Y, f: _F, f_diff: _F) -> Bool[Array, ""]:
        """TODO"""
        return _cauchy_termination(
            self.rtol, self.atol, self.norm, y, y_diff, f, f_diff
        )


CauchyTermination.__init__.__doc__ = """
TODO
"""


def _cauchy_termination(
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], Scalar],
    y: Y,
    y_diff: Y,
    f: _F,
    f_diff: _F,
) -> Bool[Array, ""]:
    y_scale = (atol + rtol * ω(y).call(jnp.abs)).ω
    f_scale = (atol + rtol * ω(f).call(jnp.abs)).ω
    y_converged = norm((ω(y_diff).call(jnp.abs) / y_scale**ω).ω) < 1
    f_converged = norm((ω(f_diff).call(jnp.abs) / f_scale**ω).ω) < 1
    return y_converged & f_converged
