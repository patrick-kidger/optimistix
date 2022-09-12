# TODO: I think the jacfwd and the jvp can probably be combined, as they both
# basically do the same thing. That might improve efficiency via parallelism.
def implicit_jvp(fn_primal, fn_rewrite, inputs, closure, linear_solver):
    """
    Takes a function `fn_primal : (inputs, closure) -> (root, residual)` and a function
    `fn_rewrite : (root, residual, inputs, closure) -> arb`.

    Has primals `fn_primal(inputs, closure)[0]` with auxiliary information
    `fn_primal(inputs, closure)[1]`.
    Has tangents `-(d(fn_rewrite)/d(root))^-1 d(fn_rewrite)/d(inputs)`, evaluated at
    `(root, residual, inputs, closure)`.

    This is used for rewriting gradients via the implicit function theorem.

    Note that due to limitations with JAX's custom autodiff, both `fn_primal` and
    `fn_rewrite` should be global functions (i.e. they should not capture any JAX array
    via closure, even if it does not participate in autodiff).
    """
    diff_args, nondiff_args = eqx.partition(inputs, eqx.is_inexact_array)
    root, residual = _implicit_backprop(
        fn_primal, fn_rewrite, nondiff_args, closure, linear_solver, diff_args
    )
    # Trim off the zero tangents we added to `residual`.
    return root, jtu.tree_map(lax.stop_gradient, residual)


@ft.partial(fixed_custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4))
def _implicit_backprop(fn_primal, fn_rewrite, nondiff_args, closure, linear_solver, diff_args):
    del fn_rewrite, linear_solver
    args = eqx.combine(diff_args, nondiff_args)
    return fn_primal(args, closure)


@_implicit_backprop.defjvp
def _implicit_backprop_jvp(
    fn_primal, fn_rewrite, nondiff_args, closure, diff_args, linear_solver, tang_diff_args
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

    operator = JacobianLinearOperator(_for_jac, _root)
    _, jvp_diff_args = jax.jvp(_for_jvp, (diff_args,), (tang_diff_args,))

    tang_root = -(linear_solve(operator, jvp_diff_args, linear_solver)**ω).ω
    tang_residual = jtu.tree_map(jnp.zeros_like, residual)
    return (root, residual), (tang_root, tang_residual)
