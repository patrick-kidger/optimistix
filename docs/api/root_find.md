# Root finding

::: optimistix.root_find

---

[`optimistix.root_find`][] supports any of the following root finders.

!!! info

    In addition to the solvers listed here, any [least squares solver](./least-squares.md) or [minimiser](./minimisers.md) may also be used as the `solver`. This is because finding the root `x` for which `f(x) = 0` can also be accomplished by finding the value `x` for which `sum(f(x)^2)` is minimised.
    
??? abstract "`optimistix.AbstractRootFinder`"

    ::: optimistix.AbstractRootFinder
        selection:
            members:
                - init
                - step
                - terminate
                - buffers

::: optimistix.Newton
    selection:
        members:
            - __init__

---

::: optimistix.Chord
    selection:
        members:
            - __init__

---

:: optimistix.Bisection
    selection:
        members:
            - __init__
