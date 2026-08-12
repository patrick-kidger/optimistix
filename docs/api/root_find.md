# Root finding

::: optimistix.root_find

---

[`optimistix.root_find`][] supports any of the following root finders.

!!! info

    In addition to the solvers listed here, any [least squares solver](./least_squares.md) or [minimiser](./minimise.md) may also be used as the `solver`. This is because finding the root `x` for which `f(x) = 0` can also be accomplished by finding the value `x` for which `sum(f(x)^2)` is minimised. If you pass in a least squares solver or minimiser, then Optimistix will automatically rewrite your problem to treat it in this way.
    
??? abstract "`optimistix.AbstractRootFinder`"

    ::: optimistix.AbstractRootFinder
        options:
            members:
                - init
                - step
                - terminate
                - postprocess

::: optimistix.Newton
    options:
        members:
            - __init__

---

::: optimistix.Chord
    options:
        members:
            - __init__

---

::: optimistix.Bisection
    options:
        members:
            - __init__

---

::: optimistix.NewtonBisection
    options:
        members:
            - __init__

---

::: optimistix.BestSoFarRootFinder
    options:
        members:
            - __init__
