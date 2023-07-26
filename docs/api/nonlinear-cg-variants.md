# Nonlinear CG methods

[`optimistix.NonlinearCG`][] supports several different methods for computing its β parameter. If you are trying multiple solvers to see which works best on your problem, then you may wish to try all four versions of nonlinear CG. These can each be passed as `NonlinearCG(..., method=...)`.

::: optimistix.polak_ribiere

::: optimistix.fletcher_reeves

::: optimistix.hestenes_stiefel

::: optimistix.dai_yuan