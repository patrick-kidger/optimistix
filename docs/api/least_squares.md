# Least squares

::: optimistix.least_squares

---

[`optimistix.least_squares`][] supports any of the following least squares solvers.

!!! info

    In addition to the solvers listed here, any [minimiser](./minimise.md) may also be used as the `solver`. This is because a least squares problem $\arg\min_\theta \sum_{i=1}^N r_i(\theta)^2$ is a special case of general minimisation problems. If you pass in a minimiser, then Optimistix will automatically treate your problem in this way.

??? abstract "`optimistix.AbstractLeastSquaresSolver`"

    ::: optimistix.AbstractLeastSquaresSolver
        options:
            members:
                - init
                - step
                - terminate
                - postprocess

??? abstract "`optimistix.AbstractGaussNewton`"

    ::: optimistix.AbstractGaussNewton
        options:
            members:
                false

::: optimistix.GaussNewton
    options:
        members:
            - __init__

---

::: optimistix.LevenbergMarquardt
    options:
        members:
            - __init__

---
            
::: optimistix.IndirectLevenbergMarquardt
    options:
        members:
            - __init__

---

::: optimistix.ScaledLevenbergMarquardt
    options:
        members:
            - __init__

[`optimistix.ScaledLevenbergMarquardt`][] supports different methods for updating the scaling operator via `ScaledLevenbergMarquardt(..., update_scaling_fn=...)`.

::: optimistix.max_scaling_update

---

::: optimistix.Dogleg
    options:
        members:
            - __init__

---

::: optimistix.BestSoFarLeastSquares
    options:
        members:
            - __init__
