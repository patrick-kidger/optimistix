# Abstract base classes

Optimistix is fully extendable. It provides a number of abstract base classes ("ABCs") which define the interfaces for custom solvers, custom line searches, etc.

**Solvers**

- Custom minimisers may be created by subclassing [`optimistix.AbstractMinimiser`][].
    - If your minimiser is derived from [Optax](https://github.com/deepmind/optax), then you can simply use [`optimistix.OptaxMinimiser`][].
- Custom least squares solvers may be created by subclassing [`optimistix.AbstractLeastSquaresSolver`][].
- Custom root finders may be created by subclassing [`optimistix.AbstractRootFinder`][].
- Custom fixed-point solvers may be created by subclassing [`optimistix.AbstractFixedPointSolver`][].

!!! info

    For creating least squares solvers like [`optimistix.GaussNewton`][] or [`optimistix.LevenbergMarquardt`][], then [`optimistix.AbstractGaussNewton`][] may be useful.
    
    For creating gradient-based minimisers like [`optimistix.BFGS`][] and [`optimistix.GradientDescent`][], then [`optimistix.AbstractGradientDescent`][] may be useful.

    In each case, they offer a general way to combine a [search and a descent](./api/searches/introduction.md).

**Line searches, trust regions, learning rates etc.**

These may be defined by subclassing [`optimistix.AbstractSearch`][]. See also the [introduction to searches and descent](./api/searches/introduction.md)

**Descent directions (steepest descent, Newton steps, Levenberg--Marquardt damped steps etc.)**

These may be defined by subclassing [`optimistix.AbstractDescent`][]. See also the [introduction to searches and descent](./api/searches/introduction.md)

**Adjoints**

These denote custom autodifferentiation strategies. These may be defined by subclassing [`optimistix.AbstractAdjoint`][].

**Norms**

Any function `PyTree -> non-negative real scalar` may be used as a norm. See also the [norms page](./api/norms.md).

**Nonlinear CG variants**

Any function `(Y, Y, Y) -> scalar` may be used to define a variant of nonlinear CG. See [`optimistix.polak_ribiere`][] as an example.
