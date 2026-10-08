# Adjoints

There are multiple ways to autodifferentiate a nonlinear solve. The two main ways are to use the implicit function theorem, and to autodifferentiate the internals of whatever solver was used.

In practice this quite an advanced API, and almost all use cases should use [`optimistix.ImplicitAdjoint`][]. (Which is the default.)

??? abstract "`optimistix.AbstractAdjoint`"

    ::: optimistix.AbstractAdjoint
        options:
            members:
                - apply

::: optimistix.ImplicitAdjoint
    options:
        members:
            - __init__

### Iterative linear solves for large fixed points

For a fixed point $y = f(y, \theta)$, implicit differentiation solves a linear system involving $I - \partial f / \partial y$. This is a separate solve from the iteration used to find the fixed point itself.

By default, `ImplicitAdjoint` uses `lineax.AutoLinearSolver`, which typically materialises the linear operator for a direct solve. For a large fixed point, you can instead pass an iterative solver such as [Lineax's GMRES](https://docs.kidger.site/lineax/api/solvers/#lineax.GMRES). It works with matrix-vector products rather than materialising the full Jacobian:

```python
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx


def total_fixed_point(theta):
    sol = optx.fixed_point(
        lambda y, args: 0.5 * y + args,
        optx.FixedPointIteration(rtol=1e-6, atol=1e-6),
        y0=jnp.zeros(3),
        args=theta,
        adjoint=optx.ImplicitAdjoint(linear_solver=lx.GMRES(rtol=1e-6, atol=1e-6)),
    )
    return jnp.sum(sol.value)


print(jax.grad(total_fixed_point)(jnp.array(1.0)))  # 6.0
```

Here each fixed-point component is $2\theta$, so the derivative of their sum is 6. The GMRES tolerances control the linear solve used for differentiation, independently of the fixed-point solver's tolerances. GMRES is an option, not a universal replacement for the default: it may converge slowly or fail, and direct solves can be faster for small problems.

---

::: optimistix.RecursiveCheckpointAdjoint
    options:
        members:
            - __init__
