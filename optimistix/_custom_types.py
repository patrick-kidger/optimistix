from collections.abc import Callable
from typing import Any, TYPE_CHECKING, TypeAlias, TypeVar

import equinox.internal as eqxi
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Real


Args: TypeAlias = Any
Aux = TypeVar("Aux")
Out = TypeVar("Out")
SolverState = TypeVar("SolverState")
SearchState = TypeVar("SearchState")
DescentState = TypeVar("DescentState")
HessianUpdateState = TypeVar("HessianUpdateState")
Y = TypeVar("Y")

Fn: TypeAlias = Callable[[Y, Args], tuple[Out, Aux]]
NoAuxFn: TypeAlias = Callable[[Y, Args], Out]
MaybeAuxFn: TypeAlias = Fn[Y, Out, Aux] | NoAuxFn[Y, Out]

sentinel: Any = eqxi.doc_repr(object(), "sentinel")


if TYPE_CHECKING:
    BoolScalarLike = bool | Array | np.ndarray
    FloatScalarLike = float | Array | np.ndarray
    IntScalarLike = int | Array | np.ndarray
    RealScalarLike = bool | int | float | Array | np.ndarray
else:
    BoolScalarLike = Bool[ArrayLike, ""]
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]
    RealScalarLike = Real[ArrayLike, ""]
