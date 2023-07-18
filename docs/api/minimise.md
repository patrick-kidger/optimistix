# Minimisation

In addition to the following, note that the [Optax]](https://github.com/deepmind/optax) library offers an extensive collection of minimisers via first-order gradient methods -- as are in widespread use for neural networks. If you would like to use these through the Optimistix API then an [`optimistix.OptaxMinimiser`][] wrapper is provided.

::: optimistix.minimise

---

[`optimistix.minimise`][] supports any of the following minimisers.

??? abstract "`optimistix.AbstractMinimiser`"

    ::: optimistix.AbstractMinimiser
        selection:
            members:
                - init
                - step
                - terminate
                - buffers

??? abstract "`optimistix.AbstractGradientDescent`"

    ::: optimistix.AbstractGradientDescent
        selection:
            members:
                false

::: optimistix.GradientDescent
    selection:
        members:
            false

---

??? abstract "`optimistix.AbstractNonlinearCG`"

    ::: optimistix.AbstractNonlinearCG
        selection:
            members:
                false

::: optimistix.NonlinearCG
    selection:
        members:
            false

---

??? abstract "`optimistix.AbstractBFGS`"

    ::: optimistix.AbstractBFGS
        selection:
            members:
                false

::: optimistix.BFGS
    selection:
        members:
            - __init__

---

::: optimistix.OptaxMinimiser
    selection:
        members:
            - __init__
