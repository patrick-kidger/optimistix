# Quadratic Programs

:::optimistix.quadratic_solve

---

[`optimistix.quadratic_solve`][] supports any of the following quadratic solvers.

??? "`optimistix.AbstractQuadraticSolver`"

    :::optimistix.AbstractQuadraticSolver
        selection:
            members:
                - init
                - step
                - terminate
                - postprocess

:::optimistix.InteriorPoint
    selection:
        members:
            - __init__