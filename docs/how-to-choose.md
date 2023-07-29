# How to choose a solver

The best choice of solver depends a lot on your problem! If you're not too sure which to pick, here is some guidance for what usually works well.

(Advanced users may note that (a) you can often adjust the behaviour of individual solvers, e.g. by using alternative line searches, and (b) that you can also mix-and-match components to derive custom optimisers, [see here](./abstract.md).)

## Minimisation problems

For relatively "well-behaved" problems -- ones where the initial guess is likely quite close to the true minimum, and the function tends to look a bit like a bowl:

<img align="center" src="./_static/quadratic_bowl.png">

then the best choice is usually an algorithm like [`optimistix.BFGS`][] or [`optimistix.NonlinearCG`][]. By assuming that your function is relatively well-behaved, then these try to take larger steps to get the minimum faster.

For "messier" problems -- where the surface of the function is not so well-behaved -- then a first-order gradient algorithm is recommend. These work by taking many small steps, and never moving too far away from the current best-so-far. As this scenario is common when training neural networks, then [neural network optimisers](https://github.com/deepmind/optax) may often successfully be repurposed for this task, so [`optimistix.OptaxMinimiser`][] is actually the recommend choice here. For example, `optimistix.OptaxMinimiser(optax.adabelief, learning_rate=1e-3, rtol=1e-8, atol=1e-8)` often works well. (Adjusting the learning rate may be necessary to get convergence that is both fast and stable.)

## Least-squares problems

These are an important special case of minimisation problems.

Either [`optimistix.LevenbergMarquardt`][] or [`optimistix.Dogleg`][] are recommended for most problems.

If your problem is particularly messy, as discussed above, then use a first-order gradient algorithm, e.g. `optimistix.OptaxMinimiser(optax.adabelief, ...)`. These are compatible with `optimistix.least_squares`.

## Root-finding problems

For one-dimensional problems, use [`optimistix.Bisection`][].

For relatively "well-behaved" problems, then either [`optimistix.Newton`][] or [`optimistix.Chord`][] are recommended.

If your problem is a little bit messy, then try [`optimistix.LevenbergMarquardt`][] or [`optimistix.Dogleg`][]. (These are compatible with root-finding problems.)

If your problem is particularly messy, as discussed above, then use a first-order gradient algorithm, e.g. `optimistix.OptaxMinimiser(optax.adabelief, ...)`. (These are also compatible with root-finding problems.)

## Fixed-point problems

For these, follow the same advice as for root-finding problems.
