"""We use notions of "searches" and "descents" to generalise line searches, trust
regions, and learning rates.

For example, the classical gradient descent algorithm is given by setting the search
to be `optimistix.LearningRate`, and setting the descent to be
`optimistix.SteepestDescent`.

As another example, Levenberg--Marquardt is given by setting the search to be
`optimistix.ClassicalTrustRegion`, and setting the descent to be
`optimistix.DampedNewtonDescent`.

As a third example, BFGS is given by setting the search to be
`optimistix.BacktrackingArmijo`, and setting the descent to be
`optimistix.NewtonDescent`.

This gives us a flexible way to mix-and-match ideas across different optimisers.

Right now, these are used exclusively for minimisers and least-squares optimisers. (As
the latter are a special case of the former, with `to_minimise = 0.5 * residuals^2`)

For an in-depth discussion of how these pieces fit together, see the documentation at
`docs/api/searches/introduction.md`.
"""

import abc
from collections.abc import Callable
from typing import Any, ClassVar, Generic, Type, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, Scalar

from ._custom_types import (
    _Iterate,
    DescentState,
    EqualityOut,
    InequalityOut,
    Out,
    SearchState,
    Y,
)
from ._misc import sum_squares, tree_dot
from ._solution import RESULTS


# TODO(jhaffner): FunctionInfo actually has its own section in the documentation, which
# should mention which flavors of FunctionInfo will carry information about constraints.
# (And bounds.)
# TODO(jhaffner): For now I'm adding constraint information where I need it. This will
# need to be systematised later.


class FunctionInfo(eqx.Module):
    """Different solvers (BFGS, Levenberg--Marquardt, ...) evaluate different
    quantities of the objective function. Some may compute gradient information,
    some may provide approximate Hessian information, etc.

    This enumeration-ish object captures the different variants.

    Available variants are
    `optimistix.FunctionInfo.{Eval, EvalGrad, EvalGradHessian, EvalGradHessianInv, Residual, ResidualJac}`.
    """  # noqa: E501

    Eval: ClassVar[type["Eval"]]
    EvalGrad: ClassVar[type["EvalGrad"]]
    EvalGradHessian: ClassVar[type["EvalGradHessian"]]
    EvalGradHessianInv: ClassVar[type["EvalGradHessianInv"]]
    Residual: ClassVar[type["Residual"]]
    ResidualJac: ClassVar[type["ResidualJac"]]

    @abc.abstractmethod
    def as_min(self) -> Scalar:
        """For a minimisation problem, returns f(y). For a least-squares problem,
        returns 0.5*residuals^2 -- i.e. its loss as a minimisation problem.
        """


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class Eval(FunctionInfo, Generic[Y, EqualityOut, InequalityOut]):
    """Has a `.f` attribute describing `fn(y)`. Used when no gradient information is
    available.
    """

    f: Scalar
    bounds: tuple[Y, Y] | None
    constraint_residual: tuple[EqualityOut, InequalityOut] | None

    def as_min(self):
        return self.f


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class EvalGrad(FunctionInfo, Generic[Y]):
    """Has a `.f` attribute as with [`optimistix.FunctionInfo.Eval`][]. Also has a
    `.grad` attribute describing `d(fn)/dy`. Used with first-order solvers for
    minimisation problems. (E.g. gradient descent; nonlinear CG.)
    """

    f: Scalar
    grad: Y
    bounds: tuple[Y, Y] | None

    def as_min(self):
        return self.f

    def compute_grad_dot(self, y: Y):
        return tree_dot(self.grad, y)


_ConstraintJacobians = (
    tuple[lx.AbstractLinearOperator, None] |
    tuple[lx.AbstractLinearOperator, lx.AbstractLinearOperator] |
    tuple[None, lx.AbstractLinearOperator] |
    None
)


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class EvalGradHessian(FunctionInfo, Generic[Y, EqualityOut, InequalityOut]):
    """Has `.f` and `.grad` attributes as with [`optimistix.FunctionInfo.EvalGrad`][].
    Also has a `.hessian` attribute describing (an approximation to) the Hessian of
    `fn` at `y`. Used with quasi-Newton minimisation algorithms, like BFGS.
    """

    f: Scalar
    grad: Y
    hessian: lx.AbstractLinearOperator
    at: Y
    bounds: tuple[Y, Y] | None

    constraint_residual: tuple[EqualityOut, InequalityOut] | None
    constraint_bounds: tuple[EqualityOut, InequalityOut] | None
    constraint_jacobians: _ConstraintJacobians

    def as_min(self):
        return self.f

    def compute_grad_dot(self, y: Y):
        return tree_dot(self.grad, y)

    def to_quadratic(self) -> Callable[[Y, Any], Scalar]:
        # With the Hessian and gradient, we have created a quadratic approximation to
        # the target function. In unconstrained optimisation, we only need to take a
        # single step, computed using a linear solve. In constrained optimisation, we
        # often need to solve the quadratic subproblem iteratively. This function allows
        # us to pass the quadratic approximation to optx.quadratic_solve, which will
        # solve the problem for the current quadratic approximation to the target
        # function and the current linear approximation of the constraints.

        # Note: since the quadratic approximation can be ASSUMED to only ever be called
        # inside a sequential quadratic program descent, we could also pass `self.at` as
        # y instead. However, this makes to many assumptions about the usage of this API
        # for my taste, which is why I opted to write this information into the function
        # information itself.

        def quadratic(_y, args):
            del args
            ydiff = (_y**ω - self.at**ω).ω
            quadratic_term = 0.5 * tree_dot(ydiff, self.hessian.mv(ydiff))
            linear_term = tree_dot(self.grad, ydiff)
            return quadratic_term + linear_term + self.f

        return quadratic

    def to_linear_constraints(
        self,
    ) -> Callable[[Y], tuple[EqualityOut, InequalityOut]] | None:
        # Creates a linear approximation to the constraints that can be used as the
        # constraint function in a quadratic subproblem. For a more detailed
        # explanation, see `.to_quadratic()`.

        # TODO: design - currently the sequential descent (imaginatively named
        # QuadraticSubproblemDescent) accepts quadratic solvers that only handle bounds.
        # We might get rid of this problem if/when we decide to remove CauchyNewton as a
        # solver, since it is terrible anyway. Until then we have this ugly workaround.
        # TODO why do I need to flag this with pyright: ignore below after gating None?
        if self.constraint_jacobians is None:
            return None
            # raise ValueError(
            #     "Cannot linearise constraints without constraint Jacobian information,
            #     "which this solver does not provide."
            # )
        else:

            def linear_constraints(_y):
                ydiff = (_y**ω - self.at**ω).ω
                ydiff_constraints = self.constraint_jac.mv(ydiff)  # pyright: ignore
                return (self.constraint_residual**ω - ydiff_constraints**ω).ω

            return linear_constraints


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class EvalGradHessianInv(FunctionInfo, Generic[Y]):
    """As [`optimistix.FunctionInfo.EvalGradHessian`][], but records the (approximate)
    inverse-Hessian instead. Has `.f` and `.grad` and `.hessian_inv` attributes.
    """

    f: Scalar
    grad: Y
    hessian_inv: lx.AbstractLinearOperator

    def as_min(self):
        return self.f

    def compute_grad_dot(self, y: Y):
        return tree_dot(self.grad, y)


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class Residual(FunctionInfo, Generic[Out]):
    """Has a `.residual` attribute describing `fn(y)`. Used with least squares problems,
    for which `fn` returns residuals.
    """

    residual: Out

    def as_min(self):
        return 0.5 * sum_squares(self.residual)


# NOT PUBLIC, despite lacking an underscore. This is so pyright gets the name right.
class ResidualJac(FunctionInfo, Generic[Y, Out]):
    """Records the Jacobian `d(fn)/dy` as a linear operator. Used for least squares
    problems, for which `fn` returns residuals. Has `.residual` and `.jac` attributes,
    where `residual = fn(y)`, `jac = d(fn)/dy`.
    """

    residual: Out
    jac: lx.AbstractLinearOperator

    def as_min(self):
        return 0.5 * sum_squares(self.residual)

    def compute_grad(self):
        conj_residual = jtu.tree_map(jnp.conj, self.residual)
        return self.jac.transpose().mv(conj_residual)

    def compute_grad_dot(self, y: Y):
        # If `self.jac` is a `lx.JacobianLinearOperator` (or a
        # `lx.FunctionLinearOperator` wrapping the result of `jax.linearize`), then
        # `min = 0.5 * residual^2`, so `grad = jac^T residual`, i.e. the gradient of
        # this. So that what we want to compute is `residual^T jac y`. Doing the
        # reduction in this order means we hit forward-mode rather than reverse-mode
        # autodiff.
        #
        # For the complex case: in this case then actually
        # `min = 0.5 * residual residual^bar`
        # which implies
        # `grad = jac^T residual^bar`
        # and thus that we want
        # `grad^T^bar y = residual^T jac^bar y = (jac y^bar)^T^bar residual`.
        # Notes:
        # (a) the `grad` derivation is not super obvious. Note that
        # `grad(z -> 0.5 z z^bar)` is `z^bar` in JAX (yes, twice the Wirtinger
        # derivative!) It uses a non-Wirtinger derivative for nonholomorphic functions:
        # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#complex-numbers-and-differentiation
        # (b) our convention is that the first term of a dot product gets the conjugate:
        # https://github.com/patrick-kidger/diffrax/pull/454#issuecomment-2210296643
        return tree_dot(self.jac.mv(jtu.tree_map(jnp.conj, y)), self.residual)


Eval.__qualname__ = "FunctionInfo.Eval"
EvalGrad.__qualname__ = "FunctionInfo.EvalGrad"
EvalGradHessian.__qualname__ = "FunctionInfo.EvalGradHessian"
EvalGradHessianInv.__qualname__ = "FunctionInfo.EvalGradHessianInv"
Residual.__qualname__ = "FunctionInfo.Residual"
ResidualJac.__qualname__ = "FunctionInfo.ResidualJac"
FunctionInfo.Eval = Eval
FunctionInfo.EvalGrad = EvalGrad
FunctionInfo.EvalGradHessian = EvalGradHessian
FunctionInfo.EvalGradHessianInv = EvalGradHessianInv
FunctionInfo.Residual = Residual
FunctionInfo.ResidualJac = ResidualJac


Eval.__init__.__doc__ = """**Arguments:**

- `f`: the scalar output of a function evaluation `fn(y)`.
"""


EvalGrad.__init__.__doc__ = """**Arguments:**

- `f`: the scalar output of a function evaluation `fn(y)`.
- `grad`: the output of a gradient evaluation `grad(fn)(y)`.
"""


EvalGradHessian.__init__.__doc__ = """**Arguments:**

- `f`: the scalar output of a function evaluation `fn(y)`.
- `grad`: the output of a gradient evaluation `grad(fn)(y)`.
- `hessian`: the output of a hessian evaluation `hessian(fn)(y)`.
"""


EvalGradHessianInv.__init__.__doc__ = """**Arguments:**

- `f`: the scalar output of a function evaluation `fn(y)`.
- `grad`: the output of a gradient evaluation `grad(fn)(y)`.
- `hessian_inv`: the matrix inverse of a hessian evaluation `(hessian(fn)(y))^{-1}`.
"""


Residual.__init__.__doc__ = """**Arguments:**

- `residual`: the vector output of a function evaluation `fn(y)`. When thought of as a
    minimisation problem the scalar value to minimise is `0.5 * residual^T residual`.
"""


ResidualJac.__init__.__doc__ = """**Arguments:**

- `residual`: the vector output of a function evaluation `fn(y)`. When thought of as a
    minimisation problem the scalar value to minimise is `0.5 * residual^T residual`.
- `jac`: the jacobian `jac(fn)(y)`.
"""


_FnInfo = TypeVar("_FnInfo", contravariant=True, bound=FunctionInfo)
_FnEvalInfo = TypeVar("_FnEvalInfo", contravariant=True, bound=FunctionInfo)


class Iterate(eqx.Module):
    """Different solvers require different types of iterates to compute the solution to
    the optimisation problem. While a solver for an unconstrained problem may only need
    access to the primal variable `y`, solving a constrained problem generally requires
    the introduction of dual variables, or Lagrange multipliers. In most constrained
    descents, these are updated alongside the primal variables (primal-dual methods).

    This enumeration-ish object captures the different variants of iterates.
    """

    Primal: ClassVar[Type["Primal"]]
    BoundedPrimal: ClassVar[Type["BoundedPrimal"]]
    AllAtOnce: ClassVar[Type["AllAtOnce"]]

    @abc.abstractmethod
    def nothing(self) -> None:
        """Dummy thing to make sure that this is an abstract class."""


class Primal(Iterate, Generic[Y], strict=True):
    """The primal variable `y`. Used in unconstrained problems."""

    y: Y

    def nothing(self):
        return None


class BoundedPrimal(Iterate, Generic[Y], strict=True):
    """The primal variable `y` and its bounds. Used in bound-constrained problems."""

    y: Y
    bounds: tuple[Y, Y] | None

    def nothing(self):
        return None


# TODO: if they all allow for things to be None, why are the "lower levels" needed at
# all?


# TODO: inequality out/ equality out can be restricted to be arrays, I think.
class AllAtOnce(Iterate, Generic[Y, EqualityOut, InequalityOut], strict=True):
    y: Y
    bounds: tuple[Y, Y] | None
    slack: InequalityOut | None
    equality_dual: EqualityOut | None
    inequality_dual: InequalityOut | None
    boundary_multipliers: tuple[Y, Y] | None

    # The primal variable y, of type Y
    # Its bounds, of type tuple[Y, Y]
    # The (primal) slack variables slack, of type InequalityOut
    # The (dual) equality multipliers equality_dual, of type EqualityOut
    # The (dual) inequality multipliers inequality_dual, of type InequalityOut
    # The (dual) boundary multipliers boundary_multipliers, of type [Y, Y]
    # Now the problem is that the bounds, equality constraints, inequality constraints,
    # or all three, may be None. # That creates 2^3 = 8 cases, too many to create their
    # own class for. So I have to allow the option for each of these attributes to be
    # None.

    # TODO: methods? Like a feasible update step size? The steps should be truncated
    # in the descent, since methods might differ (allowing for nonsynchronous stepsizes,
    # for instance, and the descent state should hold the step defined as a "full step"
    # to make sure that the search step sizes and descent step sizes remain coupled).

    # TODO: This could have a "to_barrier" method, I think. Not *quite* sure if I could
    # put everything in there, but probably quite a lot of things.

    def nothing(self):
        return None


Primal.__qualname__ = "Iterate.Primal"
BoundedPrimal.__qualname__ = "Iterate.BoundedPrimal"
AllAtOnce.__qualname__ = "Iterate.AllAtOnce"
Iterate.Primal = Primal
Iterate.BoundedPrimal = BoundedPrimal
Iterate.AllAtOnce = AllAtOnce


Primal.__init__.__doc__ = """**Arguments:**

- `y`: the (primal) optimisation variable `y`.
"""

BoundedPrimal.__init__.__doc__ = """**Arguments:**

- `y`: the (primal) optimisation variable `y`.
- `bounds`: the bounds on the optimisation variable `y`. A tuple `(lower, upper)`, where
    `lower` and `upper` must have the same PyTree structure as `y`.
"""

# TODO: AllAtOnce iterate, once this gets a proper name...
# Or make it just one iterate, especially if everything except the primal may be None.


class AbstractDescent(eqx.Module, Generic[Y, _FnInfo, DescentState], strict=True):
    """The abstract base class for descents. A descent consumes a scalar (e.g. a step
    size), and returns the `diff` to take at point `y`, so that `y + diff` is the next
    iterate in a nonlinear optimisation problem.

    See [this documentation](./introduction.md) for more information.
    """

    @abc.abstractmethod
    def init(self, y: _Iterate, f_info_struct: _FnInfo) -> DescentState:
        """Is called just once, at the very start of the entire optimisation problem.
        This is used to set up the evolving state for the descent.

        **Arguments:**

        - `y`: The initial guess for the optimisation problem.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, and potentially the gradient of `f` at `y`, etc.

        **Returns:**

        The initial state, prior to doing any descents or searches.
        """

    @abc.abstractmethod
    def query(self, y: _Iterate, f_info: _FnInfo, state: DescentState) -> DescentState:
        """Is called whenever the search decides to halt and accept its current iterate,
        and then query the descent for a new descent direction.

        This method can be used to precompute information that does not depend on the
        step size.

        For example, consider running a minimisation algorithm that performs multiple
        line searches. (E.g. nonlinear CG.) A single line search may reject many steps
        until it finds a location that it is happy with, and then accept that one. At
        that point, we must compute the direction for the next line search. That
        should be done in this method.

        **Arguments:**

        - `y`: the value of the current (just accepted) iterate.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, the gradient of `f` at `y`, etc.
        - `state`: the evolving state of the repeated descents.

        **Returns:**

        The updated descent state.
        """

    def correct(
        self, y: _Iterate, f_info: _FnInfo, state: DescentState
    ) -> DescentState:
        """A correction step, based on updated function information, to be taken when
        the search has rejected the current iterate. This is useful in constrained
        optimisation, where the constraints may be strongly nonlinear, and where the
        descent direction does not depend on the step-size (i.e. a line search method).
        With a previously factored KKT system cached in the descent state, we can solve
        for a corrected direction by updating the right-hand side with the constraint
        value at the trial iterate. For this use-case, this method may be overridden in
        the descent class.

        Note that it is generally not intended to be used with updated `expensive` parts
        of the function information, such as the Hessian of the target function, or the
        Jacobians of the constraints. Updating these then requires re-solving the linear
        system, which should be done in `query`.

        **Arguments:**

        - `y`: the value of the current (just accepted) iterate.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, the gradient of `f` at `y`, etc.
        - `state`: the evolving state of the repeated descents.

        **Returns:**

        The descent state.
        """
        del y, f_info
        return state

    @abc.abstractmethod
    def step(self, step_size: Scalar, state: DescentState) -> tuple[_Iterate, RESULTS]:
        """Computes the descent of size `step_size`.

        **Arguments:**

        - `step_size`: a non-negative scalar describing the step size to take.
        - `state`: the evolving state of the repeated descents. This will typically
            convey most of the information about the problem, having been computed in
            [`optimistix.AbstractDescent.query`][].

        **Returns:**

        A 2-tuple of `(y_diff, result)`, describing the delta to apply in
        y-space, and any error if computing the update was not successful.
        """


class AbstractSearch(eqx.Module, Generic[Y, _FnInfo, _FnEvalInfo, SearchState]):
    """The abstract base class for all searches. (Which are our generalisation of
    line searches, trust regions, and learning rates.)

    See [this documentation](./introduction.md) for more information.
    """

    @abc.abstractmethod
    def init(self, y: Y, f_info_struct: _FnInfo) -> SearchState:
        """Is called just once, at the very start of the entire optimisation problem.
        This is used to set up the evolving state for the search.

        **Arguments:**

        - `y`: The initial guess for the optimisation problem.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about the
            shape/dtype of `f` evaluated at `y`, and potentially the gradient of `f` at
            `y`, etc.

        **Returns:**

        The initial state, prior to doing any descents or searches.
        """

    @abc.abstractmethod
    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: SearchState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, SearchState]:
        """Performs a step within a search. For example, one step within a line search.

        **Arguments:**

        - `first_step`: True on the first step, then False afterwards. On the first
            step, then `y` and `f_info` will be meaningless dummy data, as no steps have
            been accepted yet. The search will normally need to special-case this case.
            On the first step, it is assumed that `accept=True` will be returned.
        - `y`: the value of the most recently "accepted" iterate, i.e. the last point at
            which the descent was `query`'d.
        - `y_eval`: the value of the most recent iterate within the search.
        - `y_diff`: equal to `y_eval - y`. Provided as a convenience, to save some
            computation, as this is commonly available from the calling solver.
        - `f_info`: An [`optimistix.FunctionInfo`][] describing information about `f`
            evaluated at `y`, the gradient of `f` at `y`, etc.
        - `f_eval_info`: An [`optimistix.FunctionInfo`][] describing information about
            `f` evaluated at `y`, the gradient of `f` at `y`, etc.
        - `state`: the evolving state of the repeated searches.

        **Returns:**

        A 4-tuple of `(step_size, accept, result, state)`, where:

        - `step_size`: the size of the next step to make. (Relative to `y`, not
            `y_eval`.) Will be passed to the descent.
        - `accept`: whether this step should be "accepted", in which case the descent
            will be queried for a new direction. Must be `True` if `first_step=True`.
        - `result`: an [`optimistix.RESULTS`][] object, for declaring any errors.
        - `state`: the updated state for the next search.
        """
