import math
from typing import Any, Callable, NewType, Optional, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, PyTree, Shaped

from ..least_squares import LeastSquaresProblem
from ..linear_operator import (
    AbstractLinearOperator,
    JacobianLinearOperator,
    PyTreeLinearOperator,
)
from ..minimise import MinimiseProblem
from ..misc import tree_inner_prod, two_norm
from ..root_find import RootFindProblem


# these are just to avoid large try-except blocks in the line search code,
# making those algorithms easier to read


def quadratic_predicted_reduction(
    gauss_newton: bool,
    diff: PyTree[Array],
    descent_state: PyTree,
    args: Any,
    options: Optional[dict[str, Any]],
):
    # The predicted reduction of a quadratic model function.
    # This is the model quadratic model function of classical trust region
    # methods localized around f(x). ie. `m(p) = g^t p + 1/2 p^T B p`
    # where `g` is the gradient, `B` the Quasi-Newton approximation to the
    # Hessian, and `p` the descent direction (diff).
    #
    # in the Gauss-Newton setting we compute
    # ```0.5 * [(Jp + r)^T (Jp + r) - r^T r]```
    # which is equivalent when `B = J^T J` and `g = J^T r`.
    if descent_state.operator is None:
        raise ValueError(
            "Cannot get predicted reduction without `operator`. "
            "`operator_inv` cannot be used with predicted_reduction."
        )
    if gauss_newton:
        rtr = two_norm(descent_state.vector) ** 2
        jacobian_term = (
            two_norm((ω(descent_state.operator.mv(diff)) + ω(descent_state.vector)).ω)
            ** 2
        )
        reduction = 0.5 * (jacobian_term - rtr)
    else:
        operator_quadratic = 0.5 * tree_inner_prod(
            diff, descent_state.operator.mv(diff)
        )
        steepest_descent = tree_inner_prod(descent_state.vector, diff)
        reduction = (operator_quadratic**ω + steepest_descent**ω).ω
    return reduction


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


class _NoAuxOut(eqx.Module):
    fn: Callable

    def __call__(self, x, args):
        f, _ = self.fn(x, args)
        return f


def compute_hess_grad(
    problem: Union[MinimiseProblem, RootFindProblem], y: PyTree[Array], args: Any
):
    jrev = jax.jacrev(problem.fn, has_aux=True)
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
