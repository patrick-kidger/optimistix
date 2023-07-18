# Fixed point solvers

::: optimistix.fixed_point

---

[`optimistix.fixed_point`][] supports any of the following fixed-point solvers.

!!! info

    In addition to the solvers listed here, all [root-finding solvers](./root_find.md) are also supported. This is because any fixed point problem "find `y` such that `y = f(y)`" can be automatically converted to a root finding problem "find `y` such that `g(y) = 0`", where `g` is defined by `g(y) = y - f(y)`.

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
