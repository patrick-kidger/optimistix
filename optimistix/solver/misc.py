import math
from typing import NewType

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PyTree, Shaped

from ..custom_types import sentinel
from ..linear_operator import (
    AbstractLinearOperator,
    JacobianLinearOperator,
    PyTreeLinearOperator,
)
from ..linear_tags import positive_semidefinite_tag, symmetric_tag


def init_derivatives(model, problem, y, needs_gradient, needs_hessian, options, args):
    #
    # This initialisation merits some explanation. Quasi-Newton and Gauss-Newton
    # methods are roughly interchangable numerically, as they both compute
    # `A diff = -y`
    # For quasi-Newton `B = A`, `y = g` where `B` is the approximate Hessian and `g`
    # the gradient. Gauss-Newton is often defined as `(J^T J) diff = J^T r`.
    # However `(J^T J)^{-1} J^T` is the pseudoinverse of `J`, i.e. the solution to a
    # linear least squares problem. Notice that Gauss-Newton is a special case of
    # Quasi-Newton with `B = J^T J` and `g = J^T r` this is why we use the terminology
    # "needs_hessian" and "needs_gradient" in both cases.
    #
    # However, using `J^T J` directly is generally not a great idea:
    # - Calling `J^T J` squares the condition number => bad for numerical stability;
    # - This can take a long time to compile: JAX isn't smart enough to treat each
    #   `J` and `J^T` together, and treats each as a separate autodiff call. (Grr,
    #   the endless problem with XLA inlining everything.)
    #
    # It's easy to avoid all of this by just solving `J diff = -r` provided the
    # linear solver used can handle the least-squares case.
    #
    # At initialization then, we can either choose to pass the pair `(B, g)` or
    # `(J, r)` and all the computations will progress more or less with no modificaiton
    # between the quasi-Newton and Guass-Newton case. This is why we choose the more
    # generic terms "operator, vector" to describe `A` and `y`.
    #
    # We also allow for the case where we would like to compute the operator
    # and/or vector in the model function itself. This happens for example in
    # Levenberg-Marquardt, where the Jacobian should have an extra `sqrt(lambda) I`
    # term appended to it.
    #

    try:
        (f_out, aux) = options["f_and_aux"]
    except KeyError:
        f_out = problem.fn(y, args)
        if problem.has_aux:
            (f_out, aux) = f_out
        else:
            aux = None

    if model.gauss_newton and not model.computes_operator:
        if needs_hessian:
            try:
                jacobian = options["jacobian"]
            except KeyError:
                jacobian = JacobianLinearOperator(
                    problem.fn, y, args, tags=problem.tags, _has_aux=True
                )
        else:
            jacobian = None

        # in the Gauss-Newton case, f_out = residuals
        return jnp.sum(f_out**2), jacobian, f_out, aux

    else:
        no_vec = not model.computes_vector
        no_op = not model.computes_operator
        if (needs_hessian and no_op) or (needs_gradient and no_vec):
            try:
                gradient = options["gradient"]
            except KeyError:
                gradient = sentinel
            try:
                hessian = options["hessian"]
            except KeyError:
                hessian = sentinel
            if (hessian == sentinel and needs_hessian) or (
                gradient == sentinel and needs_gradient
            ):
                jrev = jax.jacrev(problem.fn, has_aux=problem.has_aux)

                if gradient == sentinel and needs_gradient and no_vec:
                    gradient = jrev(y, args)
                    if problem.has_aux:
                        (gradient, _) = gradient
                elif gradient == sentinel:
                    gradient = None

                if hessian == sentinel and needs_hessian and no_op:
                    struct_helper = jrev(y, args)
                    hessian = jax.jacfwd(jrev, has_aux=problem.has_aux)(y, args)
                    if problem.has_aux:
                        (struct_helper, _) = struct_helper
                        (hessian, _) = hessian
                    hessian = PyTreeLinearOperator(
                        hessian,
                        jax.eval_shape(lambda: struct_helper),
                        tags={positive_semidefinite_tag, symmetric_tag},
                    )
                elif hessian == sentinel:
                    hessian = None
        else:
            hessian = gradient = None

        return f_out, gradient, hessian, aux


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
