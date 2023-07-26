# Least squares

::: optimistix.least_squares

---

[`optimistix.least_squares`][] supports any of the following least squares solvers.

!!! info

    In addition to the solvers listed here, any [minimiser](./minimisers.md) may also be used as the `solver`. This is because a least squares problem $\argmin_\theta \sum_{i=1}^N r_i(\theta)^2$ is a special case of general minimisation problems. If you pass in a minimiser, then Optimistix will automatically treate your problem in this way.

??? abstract "`optimistix.AbstractLeastSquaresSolver`"

    ::: optimistix.AbstractLeastSquaresSolver
        selection:
            members:
                - init
                - step
                - terminate
                - buffers

??? abstract "`optimistix.AbstractGaussNewton`"

    ::: optimistix.AbstractGaussNewton
        selection:
            members:
                false

::: optimistix.GaussNewton
    selection:
        members:
            - __init__

---

::: optimistix.LevenbergMarquardt
    selection:
        members:
            - __init__

---
            
::: optimistix.IndirectLevenbergMarquardt
    selection:
        members:
            - __init__

---

::: optimistix.Dogleg
    selection:
        members:
            - __init__