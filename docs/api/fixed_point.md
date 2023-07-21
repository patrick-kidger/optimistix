# Fixed point solvers

::: optimistix.fixed_point

---

[`optimistix.fixed_point`][] supports any of the following fixed-point solvers.

!!! info

    In addition to the solvers listed here, any [root finder](./root-find.md) may also be used as the `solver`. This is because finding the fixed point `x` for which `f(x) = x`, can also be accomplished by finding the root `x` for which `f(x) - x = 0`.

    Likewise, any [least squares solver](./least-squares.md) or [minimiser](./minimisers.md) may also be used as the `solver`. This is because finding the root `x` for which `f(x) = x` can also be accomplished by finding the value `x` for which `sum((f(x) - x)^2)` is minimised.

??? abstract "`optimistix.AbstractFixedPointSolver`"

    ::: optimistix.AbstractFixedPointSolver
        selection:
            members:
                - init
                - step
                - terminate
                - buffers

::: optimistix.FixedPointIteration
    selection:
        members:
            - __init__
