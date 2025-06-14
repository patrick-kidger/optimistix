from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar

import equinox.internal as eqxi


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
