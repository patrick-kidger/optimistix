from typing import Any, Sequence, TYPE_CHECKING, Union

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Fn, LineSearchAux
from .._line_search import AbstractLineSearch
from .._solution import RESULTS


if TYPE_CHECKING:
    _Node = Any
else:
    _Node = eqxi.doc_repr(Any, "Node")


class LearningRate(AbstractLineSearch[Bool[Array, ""]]):
    learning_rate: Scalar

    def first_init(
        self,
        vector: PyTree[Array],
        operator: lx.AbstractLinearOperator,
        options: dict[str, Any],
    ) -> Scalar:
        return self.learning_rate

    def init(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> Bool[Array, ""]:
        # Needs to run once, so set the state to be whether or not
        # it has ran yet.
        return jnp.array(False)

    def step(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        state: Bool[Array, ""],
        tags: frozenset[object],
    ) -> tuple[Scalar, Bool[Array, ""], LineSearchAux]:
        (f_val, (_, diff, aux, result, _)) = fn(y, args)
        return (
            y,
            jnp.array(True),
            (f_val, diff, aux, result, self.learning_rate),
        )

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, LineSearchAux],
        y: Scalar,
        args: Any,
        options: dict[str, Any],
        state: Bool[Array, ""],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state, RESULTS.successful

    def buffers(self, state: Bool[Array, ""]) -> Union[_Node, Sequence[_Node]]:
        return ()
