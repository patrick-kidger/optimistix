# Line searches

The main entry point for line searches is the [`optimistix.line_search`][] function. In practice this is rarely used unless you are defining your own solver. See below for a list of line searches, which can be passed to other solvers. (E.g. [`optimistix.BFGS`][] requires a line search.)

::: optimistix.line_search

---

??? abstract "`optimistix.AbstractLineSearch`"

    ::: optimistix.AbstractLineSearch
        selection:
            members:
                false

::: optimistix.LearningRate
    selection:
        members:
            - __init__

---

::: optimistix.BacktrackingArmijo
    selection:
        members:
            - __init__

---

::: optimistix.ClassicalTrustRegion
    selection:
        members:
            - __init__

---

::: optimistix.LinearTrustRegion
    selection:
        members:
            - __init__
