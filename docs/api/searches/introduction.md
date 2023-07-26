# Introduction

Optimistix generalises line searches, trust regions, and learning rates. This generalised approach uses two notions: a "descent", and a "search".

This is an advanced API for building minimisers and least-squares solvers.

## The abstractions

Suppose first we have a function $f \colon \mathbb{R}^n \to \mathbb{R}$, that we would like to minimise. (We'll come back to least-squares problems in a bit.) Suppose we have the current iterate $y_n$, and have already evaluated quantities like the value $f_n = f(y_n)$, the gradient $g_n = \nabla f(y_n)$, and potentially also quantites like a Hessian approximation $H_n \approx \nabla^2 f(y_n)$ or an inverse-Hessian approximation $V_n \approx (\nabla^2 f(y_n))^{-1}$.

Then a "descent" is a function $D(\,\cdot\,, f_n, g_n, H_n, V_n) \colon (0, \infty) \to \mathbb{R}^n$. This defines what it means to make an update "of size $t$". For example, simple steepest descent is defined by $D(t, f_n, g_n, H_n, V_n) = -t g_n$. This is an update "of size $t$" directly downhill.

Meanwhile, the "search" is a procedure for choosing the scalar parameter $t$. This is a routine $S(f, D, y_n, g_n, H_n, V_n) \to (0, \infty)$. We might typically think of this as being a line search, but it can be other things too. For example, we can represent a learning rate $\alpha$ as the constant function $S(f, y_n, g_n, H_n, V_n) = \alpha$. Note that $S$ is a function of both the descent $D$ and the target function $f$, as it may make queries to them, in order to choose the appropriate step size.

In addition, we allow the descent and the search to have some state $d_n$ and $s_n$ respectively, that they can use to pass information to themself between iterates. We elided this in our notation above, but introducing it now, we iterate:

$$ t_{n+1}, s_{n+1} = S(f, D, y_n, g_n, H_n, V_n, s_n, d_n) $$

$$ \delta_{n+1}, d_{n+1} = D(t_{n+1}, f_n, g_n, H_n, V_n, d_n) $$

$$ y_{n+1} = y_n + \delta_n $$

So the search picks a scalar parameter, and the descent decides how big an update that corresponds to.

## Examples

Alright, so far this probably feels a bit abstract, and maybe a bit too generic to be useful. Let's look at some examples.

- Gradient descent is obtained by combining steepest descent $D(\cdots) = - t g_n$ with a fixed learning rate $S(\cdots) = \alpha$.
- Levenberg--Marquardt is obtained by combining damped Newton descent $D(\cdots) = -(H_n + t I)^{-1} g_n$ with a trust region algorithm for $S$. The scalar $t$ describes the radius of the trust region, which is variably increased and decreased to try and stay within tolerances.
- BFGS is obtained by combining a Newton descent $D(\cdots) = -V_n g_n$ with a backtracking Armijo line search for $S$.
- etc.

And the really cool thing about this is how these ideas can now be mix-and-match'd! For example, this:
```python
import optimistix as optx

optx.BFGS(descent=optx.DoglegDescent(), search=optx.BacktrackingArmijo())
```
will, at each step:

- form a quadratic approximation to the target function, using the approximate Hessian that is iteratively built up by the BFGS algorithm;
- with a piecwise-linear dogleg-shaped descent path (interpolating between the steepest descent and Newton descent directions);
- and decides how big a step it should make down that (nonlinear) path by performing a backtracking search.

Moreover, it can be used to solve either minimisation or least squares problems.

These abstractions make it possible to build very flexible optimisers, to try out whatever works best on your problem.

!!! info

    Optimistix makes heavy use of these abstractions internally. For example, take a look at the source code for `optimixis.LevenbergMarquardt`: it works by subclassing [`optimistix.AbstractGaussNewton`][] (which provides the overall strategy), and then setting a choice of descent and search. You can define [custom solvers](../abstract.md) in exactly the same way.

## Least-squares problems

Finally, how are least-squares problems handled?

Recall that a least-squares problem has a function $r(y)$ that outputs a vector of residuals, for which the goal is to try and find the $y$ minimising $m(y) = \sum_i 0.5 r(y)_i^2$. This is instrinsically already a minimisation problem. When we differentiate $\frac{\mathrm{d}m}{\mathrm{d}y_j} = \sum_i r(y)_i \frac{\mathrm{d}r_i}{\mathrm{d}y_j}$, which can be computed efficiently as a vector-Jacobian product. This gives us $g_n$ from above.

Meanwhile, for the Hessian $\nabla^2 m$ we make the Gauss--Newton approximation $\sum_k \frac{\mathrm{d}r_k}{\mathrm{d}y_i}\frac{\mathrm{d}r_k}{\mathrm{d}y_j}$. (Exercise for the reader: derive this.) This gives us $H_n$ from above.

Meanwhile, we can get $V_n$ by inverting $H_n$, if necessary.
