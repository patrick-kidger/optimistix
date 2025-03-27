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

---

::: optimistix.RecursiveCheckpointAdjoint
    options:
        members:
            - __init__
