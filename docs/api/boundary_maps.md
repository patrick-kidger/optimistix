# Boundary Maps

Boundary maps compute a projection of a current iterate `y` onto a feasible set if `y` 
is not in the feasible set, and otherwise return it unchanged. These can be composed 
with a gradient-based minimiser using [`optimistix.ProjectedGradientDescent`][].

!!! info
    For feasible sets with a simple geometry, the projection has an analytic solution 
    and is cheap to evaluate. However, since the projection necessarily occurs after the
    step has been computed, constraints and bounds are not taken into account when 
    computing the step, or deciding whether to terminate the solve.
    If interactions between the target and constraint functions are expected to be 
    complex and substantially affect the convergence properties of the solve, a natively
    constrained solver such as [`optimistix.IPOPTLike`][] is the preferred choice.


??? abstract "`AbstractBoundaryMap`"

    ::: optimistix.AbstractBoundaryMap
        selection:
            members:
                - __call__


::: optimistix.BoxProjection
    selection:
        members:
            False

---