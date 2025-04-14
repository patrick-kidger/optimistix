# Minimisation

In addition to the following, note that the [Optax](https://github.com/deepmind/optax) library offers an extensive collection of minimisers via first-order gradient methods -- as are in widespread use for neural networks. If you would like to use these through the Optimistix API then an [`optimistix.OptaxMinimiser`][] wrapper is provided.

::: optimistix.minimise

---

[`optimistix.minimise`][] supports any of the following minimisers.

??? abstract "`optimistix.AbstractMinimiser`"

    ::: optimistix.AbstractMinimiser
        options:
            members:
                - init
                - step
                - terminate
                - postprocess

??? abstract "`optimistix.AbstractGradientDescent`"

    ::: optimistix.AbstractGradientDescent
        options:
            members:
                false

::: optimistix.GradientDescent
    options:
        members:
            - __init__

---

??? abstract "`optimistix.AbstractQuasiNewton`"

    ::: optimistix.AbstractQuasiNewton
        options:
            members:
                false

    [`optimistix.AbstractQuasiNewton`][] supports several update methods for the Hessian approximation. Optimistix currently provides two of these ([`optimistix.BFGSUpdate`][] and [`optimistix.DFPUpdate`][]), but you can also implement your own using the [`optimistix.AbstractQuasiNewtonUpdate`][] interface.

    ::: optimistix.AbstractQuasiNewtonUpdate
        options:
            members:
                - __call__

    ::: optimistix.BFGSUpdate
        options:
            members:
                - __init__

    ::: optimistix.DFPUpdate
        options:
            members:
                - __init__

::: optimistix.BFGS
    options:
        members:
            - __init__

::: optimistix.DFP
    options:
        members:
            - __init__

---

::: optimistix.OptaxMinimiser
    options:
        members:
            - __init__

`optim` in [`optimistix.OptaxMinimiser`][] is an instance of an Optax minimiser. For example, correct usage is `optimistix.OptaxMinimiser(optax.adam(...), ...)`, not `optimistix.OptaxMinimiser(optax.adam, ...)`.

---

::: optimistix.NonlinearCG
    options:
        members:
            - __init__

[`optimistix.NonlinearCG`][] supports several different methods for computing its β parameter. If you are trying multiple solvers to see which works best on your problem, then you may wish to try all four versions of nonlinear CG. These can each be passed as `NonlinearCG(..., method=...)`.

::: optimistix.polak_ribiere

::: optimistix.fletcher_reeves

::: optimistix.hestenes_stiefel

::: optimistix.dai_yuan

---

::: optimistix.NelderMead
    options:
        members:
            - __init__

---

::: optimistix.BestSoFarMinimiser
    options:
        members:
            - __init__
