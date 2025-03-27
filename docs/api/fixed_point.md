# Fixed points

::: optimistix.fixed_point

---

[`optimistix.fixed_point`][] supports any of the following fixed-point solvers.

!!! info

    In addition to the solvers listed here, any [root finder](./root_find.md) may also be used as the `solver`. This is because finding the fixed point `x` for which `f(x) = x`, can also be accomplished by finding the root `x` for which `f(x) - x = 0`. If you pass in a root finder, then Optimistix will automatically rewrite your problem to treat it in this way.

    Likewise, any [least squares solver](./least_squares.md) or [minimiser](./minimise.md) may also be used as the `solver`. This is because finding the root `x` for which `f(x) = x` can also be accomplished by finding the value `x` for which `sum((f(x) - x)^2)` is minimised. If you pass in a least squares solver or minimiser, then Optimistix will automatically rewrite your problem to treat it in this way.

??? abstract "`optimistix.AbstractFixedPointSolver`"

    ::: optimistix.AbstractFixedPointSolver
        options:
            members:
                - init
                - step
                - terminate
                - postprocess

::: optimistix.FixedPointIteration
    options:
        members:
            - __init__

---

::: optimistix.BestSoFarFixedPoint
    options:
        members:
            - __init__
