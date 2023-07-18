# Least squares

::: optimistix.least_squares

---

[`optimistix.least_squares`][] supports any of the following least squares solvers.

!!! info

    In addition to the solvers listed here, all [minimisers](./minimisers.md). This is because a least squares problem $\argmin_\theta \sum_{i=1}^N r_i(\theta)^2$ can be immediately interpreted as a minimisation problem.

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