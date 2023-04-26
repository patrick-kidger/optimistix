from typing import Any, Callable, FrozenSet

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import PyTree

from .linear_operator import JacobianLinearOperator
from .linear_solve import AbstractLinearSolver, linear_solve


def implicit_jvp(
    fn_primal: Callable,
    fn_rewrite: Callable,
    inputs: PyTree,
    closure: Any,
    tags: FrozenSet[object],
    linear_solver: AbstractLinearSolver,
):
    """Rewrites gradients via the implicit function theorem.

    **Arguments:**

    - `fn_primal` is a function `(inputs, closure) -> (root, residual)`.
    - `fn_rewrite` is a function `(root, residual, inputs, closure) -> arbitrary`.
    - `inputs` is some input PyTree of the primal inputs to the computation.
    - `closure` is an arbitrary Python object, used to pass auxiliary inputs to the
        computation.
    - `tags`: any tags (symmetric, diagonal, ...) for the matrix
        `d(fn_rewrite)/d(root)`.
    - `linear_solver`: an `AbstractLinearSolver`, used to solve the linear problem
        on the backward pass.

    Note that due to limitations with JAX's custom autodiff, both `fn_primal` and
    `fn_rewrite` should be global functions (i.e. they should not capture any JAX array
    via closure, even if it does not participate in autodiff).

    **Returns:**

    This function returns `fn_primal(inputs, closure)`. The first output is the output
    primal, whilst the second is auxiliary information.

    The primals have tangents `-(d(fn_rewrite)/d(root))^-1 d(fn_rewrite)/d(inputs)`,
    evaluated at `(root, residual, inputs, closure)`.
    """
    root, residual = _implicit_impl(
        fn_primal, fn_rewrite, inputs, closure, tags, linear_solver
    )
    return root, jtu.tree_map(eqxi.nondifferentiable_backward, residual)


@eqx.filter_custom_jvp
def _implicit_impl(fn_primal, fn_rewrite, inputs, closure, tags, linear_solver):
    del fn_rewrite, tags, linear_solver
    return fn_primal(inputs, closure)


def _is_none(x):
    return x is None


def _for_jac(root, args):
    fn_rewrite, residual, inputs, closure = args
    return fn_rewrite(root, residual, inputs, closure)


@_implicit_impl.defjvp
def _implicit_impl_jvp(primals, tangents):
    fn_primal, fn_rewrite, inputs, closure, tags, linear_solver = primals
    (
        t_fn_primal,
        t_fn_rewrite,
        t_inputs,
        t_closure,
        t_tags,
        t_linear_solver,
    ) = tangents

    def _assert_false(x):
        assert False

    jtu.tree_map(
        _assert_false, (t_fn_primal, t_fn_rewrite, t_closure, t_tags, t_linear_solver)
    )
    del t_fn_primal, t_fn_rewrite, t_closure, t_tags, t_linear_solver
    no_tangent = jtu.tree_map(_is_none, t_inputs, is_leaf=_is_none)
    nondiff, diff = eqx.partition(inputs, no_tangent, is_leaf=_is_none)

    root, residual = implicit_jvp(
        fn_primal, fn_rewrite, inputs, closure, tags, linear_solver
    )

    def _for_jvp(_diff):
        _inputs = eqx.combine(_diff, nondiff)
        return fn_rewrite(root, residual, _inputs, closure)

    operator = JacobianLinearOperator(
        _for_jac, root, (fn_rewrite, residual, inputs, closure), tags=tags
    )
    _, jvp_diff = jax.jvp(_for_jvp, (diff,), (t_inputs,))

    t_root = linear_solve(operator, jvp_diff, linear_solver)
    t_root_value = (-ω(t_root.value)).ω
    t_root = eqx.tree_at(lambda x: x.value, t_root, t_root_value)

    t_residual = jtu.tree_map(lambda _: None, residual)
    return (root, residual), (t_root, t_residual)
