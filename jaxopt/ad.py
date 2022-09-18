# TODO: I think the jacfwd and the jvp can probably be combined, as they both
# basically do the same thing. That might improve efficiency via parallelism.
def implicit_jvp(
    fn_primal: Callable,
    fn_rewrite: Callable,
    inputs: PyTree[Array],
    closure: Any,
    pattern: Pattern,
    linear_solver: AbstractLinearSolver:
):
    """Rewrites gradients via the implicit function theorem.

    **Arguments:**

    - `fn_primal` is a function `(inputs, closure) -> (root, residual)`.
    - `fn_rewrite` is a function `(root, residual, inputs, closure) -> arbitrary`.
    - `inputs` is some PyTree of arrays, which are the primal inputs to the
        computation.
    - `closure` is an arbitrary Python object, used to pass auxiliary inputs to the
        computation.
    - `pattern`: any pattern (symmetric, diagonal, ...) in `d(fn_rewrite)/d(root)`.
        As `AbstractLinearOperator`.
    - `linear_solver` is an `AbstractLinearSolver`, used to solve the linear problem
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
    diff_args, nondiff_args = eqx.partition(inputs, eqx.is_inexact_array)
    root, residual = _implicit_impl(
        fn_primal, jac_rewrite, nondiff_args, closure, pattern, linear_solver, diff_args
    )
    # Trim off the zero tangents we added to `residual`.
    return root, jtu.tree_map(lax.stop_gradient, residual)


@ft.partial(fixed_custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def _implicit_impl(fn_primal, fn_rewrite, nondiff_args, closure, pattern, linear_solver, diff_args):
    del fn_rewrite, pattern, linear_solver
    args = eqx.combine(diff_args, nondiff_args)
    return fn_primal(args, closure)


@_implicit_impl.defjvp
def _implicit_impl_jvp(
    fn_primal, fn_rewrite, nondiff_args, closure, pattern, linear_solver, diff_args, tang_diff_args
):
    (diff_args,) = diff_args
    (tang_diff_args,) = tang_diff_args
    root, residual = _implicit_backprop(
        fn_primal, fn_rewrite, nondiff_args, closure, linear_solver, diff_args
    )

    def _for_jac(_root, _):
        args = eqx.combine(nondiff_args, diff_args)
        return fn_rewrite(_root, residual, args, closure)

    def _for_jvp(_diff_args):
        _args = eqx.combine(nondiff_args, _diff_args)
        return fn_rewrite(root, residual, _args, closure)

    operator = JacobianLinearOperator(_for_jac, _root, pattern=pattern)
    _, jvp_diff_args = jax.jvp(_for_jvp, (diff_args,), (tang_diff_args,))

    tang_root = (-linear_solve(operator, jvp_diff_args, linear_solver)**ω).ω
    tang_residual = jtu.tree_map(jnp.zeros_like, residual)
    return (root, residual), (tang_root, tang_residual)
