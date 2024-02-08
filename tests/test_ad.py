import jax
import lineax as lx
import optimistix.internal as optxi


def test_residual_nonarray_no_jit():
    def primal(inputs):
        return (inputs - 2) ** 2, 5

    def rewrite(root, residual, inputs):
        return root**2 + inputs**2

    @jax.grad
    def run(x):
        sol, aux = optxi.implicit_jvp(
            primal, rewrite, x, tags=frozenset(), linear_solver=lx.LU()
        )
        return sol + aux

    run(4.0)
