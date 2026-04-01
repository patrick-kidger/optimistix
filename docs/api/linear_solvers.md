# Linear solvers

[`optimistix.NewtonDescent`][], [`optimistix.IndirectDampedNewtonDescent`][], and related descents accept a `linear_solver` argument. The following solver is provided by Optimistix; any `lineax.AbstractLinearSolver` may also be used.

::: optimistix.TruncatedCG
    options:
        members:
            - __init__
