import math
from typing import Any, Callable, NewType

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree, Scalar, Shaped

from ..least_squares import LeastSquaresProblem
from ..linear_operator import (
    AbstractLinearOperator,
    JacobianLinearOperator,
    PyTreeLinearOperator,
)
from ..minimise import MinimiseProblem
from ..root_find import RootFindProblem


# these are just to avoid large try-except blocks in the line search code,
# making those algorithms easier to read


def get_vector_operator(options):
    try:
        vector = options["vector"]
    except KeyError:
        raise ValueError("vector must be passed vector " "via `options['vector']`")
    try:
        operator = options["operator"]
    except KeyError:
        raise ValueError(
            "operator must be passed operator " "via `options['operator']`"
        )
    return vector, operator


def get_f0(
    fn: Callable[[Scalar, Any], PyTree], options: dict[str, Any]
) -> tuple[Float[ArrayLike, ""], Bool[Array, ""]]:
    # WARNING: this will not work with generic fn
    try:
        f0 = options["f0"]
        compute_f0 = options["compute_f0"]
    except KeyError:
        f0, *_ = jtu.tree_map(
            lambda x: jnp.zeros(shape=x.shape), jax.eval_shape(fn, 0.0, None)
        )
        compute_f0 = jnp.array(True)
    return f0, compute_f0


class _NoAuxOut(eqx.Module):
    fn: Callable

    def __call__(self, x, args):
        f, _ = self.fn(x, args)
        return f


def compute_hess_grad(
    problem: MinimiseProblem | RootFindProblem, y: PyTree[Array], args: Any
):
    jrev = jax.jacrev(problem.fn, has_aux=problem.has_aux)
    grad, aux = jrev(y, args)
    hessian, _ = jax.jacfwd(jrev, has_aux=True)(y, args)
    hessian = PyTreeLinearOperator(
        hessian,
        output_structure=jax.eval_shape(lambda: grad),
    )
    return grad, hessian, aux


def compute_jac_residual(problem: LeastSquaresProblem, y: PyTree[Array], args: Any):
    residual, aux = problem.fn(y, args)
    problem.tags
    jacobian = JacobianLinearOperator(
        problem.fn, y, args, tags=problem.tags, _has_aux=True
    )
    return residual, jacobian, aux


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
