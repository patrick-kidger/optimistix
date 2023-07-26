# Norms

At several points, it is necessary to compute the norm of some arbitrary PyTree. For example, to check that the difference between two adjacent iterates is small: `||y_{n+1} - y_n|| < ε`, and thus that an optimisation algorithm has converged.

Optimistix includes the following norms. (But any function `PyTree -> non-negative real scalar` will suffice.)

!!! info

    For the curious: Optimistix typically uses [`optimistix.max_norm`][] as its default norm throughout. This is because it is invariant to the size of the problem, i.e. adding extra zero padding will never affect its output. This helps to ensure a consistent experience as the problem size changes. (This property is "sort of" true of [`optimistix.rms_norm`][], and not at all true of [`optimistix.two_norm`][].)

::: optimistix.max_norm

::: optimistix.rms_norm

::: optimistix.two_norm
