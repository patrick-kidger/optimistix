# Adjoints

There are multiple ways to autodifferentiate a nonlinear solve. The two main ways are to use the implicit function theorem, and to autodifferentiate the internals of whatever solver was used.

In practice this quite an advanced API, and almost all use cases should use [`optimistix.ImplicitAdjoint`][]. (Which is the default.)

??? abstract "`optimistix.AbstractAdjoint`"

    ::: optimistix.AbstractAdjoint
        selection:
            members:
                - apply

::: optimistix.ImplicitAdjoint
    selection:
        members:
            - __init__

---

::: optimistix.RecursiveCheckpointAdjoint
    selection:
        members:
            - __init__
