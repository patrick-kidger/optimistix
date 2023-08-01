# Introduction

This is an advanced API for minimisers and least-squares solvers.

Optimistix has a generalised approach to line searches, trust regions, and learning rates. This generalised approach uses two notions: a "search", and a "descent".

**Searches**

Consider a function $f \colon \mathbb{R}^n \to \mathbb{R}$, that we would like to minimise. Searches consume information about values+gradients+Hessians etc of $f$, and produce a single scalar value. This corresponds to the distance along a line search, the radius of a trust region, or the value of a learning rate.

**Descents**

Descents consume information about values+gradients+Hessians etc. of $f$, along with this scalar value, and give the size of the update to make. This corresponds to `-scalar * gradient` for gradient descent, `-scalar * (Hessian^{-1} gradient)` for (Gauss--)Newton algorithms, the distance along the dogleg path with Dogleg, `(Jacobian^T Jacobian + scalar * I)^{-1} gradient` for Levenberg--Marquardt (damped Newton), and so on. [Although `Jacobian^T Jacobian` isn't actually materialised -- the implementation does the smart thing and solves a least-squares problem using QR.]

**Examples**

- Gradient descent is obtained by combining a fixed learning rate with steepest descent.
- Levenberg--Marquardt is obtained by combining a trust region algorithm with a damped Newton descent.
- BFGS is obtained by combining a backtracking Armijo line search with a Newton descent.
- etc.

**Acceptance/rejection**

The search is evaluated on every step of the solver. The descent is only evaluated on some steps.

Consider performing a backtracking search along a Newton descent direction. In this case, we don't need to re-solve the linear system `Hessian^{-1} gradient` at every step -- we typically fix these "for the duration of" the search.

Quote marks used above because we don't make a distinction between the steps of the overall optimiser, and the steps within e.g. a line search: they are flattened into a single loop. This ends up being an easier abstraction to work in generality, and a useful performance optimisation when handling batches of data.

Thus, we refer to "accepted" steps as being those at which we re-evaluate the descent, and "rejected" steps as being the others. The search gets to decided which steps are accepted and rejected. For example, learning rate accepts every step, whilst a backtracking search accepts those steps with a good enough improvement (those satisfying the Armijo condition).

**Minimisation vs least-squares**

- _Minimisation_

    For a minimisation problem "minimise $f \colon \mathbb{R}^n -> \mathbb{R}$", then the quantities that might be evaluated are typically evaluations $f(y)$, gradients $\nabla f(y)$, and (approximations to) the Hessian $\nabla^2 f(y)$. 

- _Least squares_

    For a least-squares problem, we typically start with a function producing residuals $r \colon \mathbb{R}^n \to \mathbb{R}^m$, and seek to minimise $0.5 \sum_i r(y)_i^2$.

    This is simply a minimisation problem for $f(y) = 0.5 \sum_i r(y)_i^2$. Evaluations $f(y)$ can be obtained directly. Gradients $\nabla f(y) = r(y) \nabla r(y)$, which can be computed efficiently as a vector-Jacobian product. Hessians may be approximated via the Gauss--Newton approximation $\nabla^2 f(y) \approx (\nabla r(y))^{T} (\nabla r(y))$.

**API**

All searches inherit from [`optimistix.AbstractSearch`][], and all descents inherit from [`optimistix.AbstractDescent`][]. See the [searches](./searches.md) and [descents](./descents.md) pages.

The varying evaluation/gradient/Hessian/Jacobian information is passed to these as an [`optimistix.FunctionInfo`][]. See the [function info](./function_info.md) page.

**Custom solvers**

Finally, the really cool thing about these abstractions is how these can now be mix-and-match'd! For example,
```python
from collections.abc import Callable
import optimistix as optx

class HybridSolver(optx.AbstractBFGS):
    rtol: float
    atol: float
    norm: Callable
    use_inverse: bool = True
    descent: optx.AbstractDescent = optx.DoglegDescent()
    search: optx.AbstractSearch = optx.LearningRate(0.1)
```
will at each step:

- form a quadratic approximation to the target function, using the approximate Hessian that is iteratively built up by the BFGS algorithm;
- then build a piecwise-linear dogleg-shaped descent path (interpolating between the steepest descent and Newton descent directions);
- make a fixed step of length `0.1` down the length of this path.

Moreover, this can be used to solve either minimisation or least squares problems.

As such these abstractions make it possible to build very flexible optimisers, to try out whatever works best on your problem.

!!! info

    Optimistix makes heavy use of these abstractions internally. For example, take a look at the source code for `optimixis.LevenbergMarquardt`: it works by subclassing [`optimistix.AbstractGaussNewton`][] (which provides the overall strategy), and then setting a choice of descent and search. You can define [custom solvers](../../abstract.md) in exactly the same way.
