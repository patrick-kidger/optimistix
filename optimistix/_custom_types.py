from typing import Any, Callable, TypeVar
from typing_extensions import TypeAlias

import equinox.internal as eqxi
from jaxtyping import Array, PyTree, Scalar

from ._solution import RESULTS


Args: TypeAlias = Any
Aux = TypeVar("Aux")
Out = TypeVar("Out")
SolverState = TypeVar("SolverState")
Y = TypeVar("Y")

Fn: TypeAlias = Callable[[Y, Args], tuple[Out, Aux]]
LineSearchAux: TypeAlias = tuple[Scalar, PyTree[Array], PyTree[Array], RESULTS, Scalar]

sentinel: Any = eqxi.doc_repr(object(), "sentinel")
