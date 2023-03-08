import math
from typing import NewType

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PyTree, Shaped

from ..linear_operator import AbstractLinearOperator


PackedStructures = NewType("PackedStructures", eqxi.Static)


def pack_structures(operator: AbstractLinearOperator) -> PackedStructures:
    structures = operator.out_structure(), operator.in_structure()
    leaves, treedef = jtu.tree_flatten(structures)  # handle nonhashable pytrees
    return PackedStructures(eqxi.Static((leaves, treedef)))


def ravel_vector(
    pytree: PyTree[Array], packed_structures: PackedStructures
) -> Shaped[Array, " size"]:
    leaves, treedef = packed_structures.value
    out_structure, _ = jtu.tree_unflatten(treedef, leaves)
    # `is` in case `tree_equal` returns a Tracer.
    if eqx.tree_equal(jax.eval_shape(lambda: pytree), out_structure) is not True:
        raise ValueError("pytree does not match out_structure")
    # not using `ravel_pytree` as that doesn't come with guarantees about order
    leaves = jtu.tree_leaves(pytree)
    dtype = jnp.result_type(*leaves)
    return jnp.concatenate([x.astype(dtype).reshape(-1) for x in leaves])


def unravel_solution(
    solution: Shaped[Array, " size"], packed_structures: PackedStructures
) -> PyTree[Array]:
    leaves, treedef = packed_structures.value
    _, in_structure = jtu.tree_unflatten(treedef, leaves)
    leaves, treedef = jtu.tree_flatten(in_structure)
    sizes = np.cumsum([math.prod(x.shape) for x in leaves[:-1]])
    split = jnp.split(solution, sizes)
    assert len(split) == len(leaves)
    shaped = [x.reshape(y.shape).astype(y.dtype) for x, y in zip(split, leaves)]
    return jtu.tree_unflatten(treedef, shaped)


def transpose_packed_structures(
    packed_structures: PackedStructures,
) -> PackedStructures:
    leaves, treedef = packed_structures.value
    out_structure, in_structure = jtu.tree_unflatten(treedef, leaves)
    leaves, treedef = jtu.tree_flatten((in_structure, out_structure))
    return eqxi.Static((leaves, treedef))
