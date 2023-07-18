# Root finding

::: optimistix.root_find

---

[`optimistix.root_find`][] supports any of the following root finders.

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
