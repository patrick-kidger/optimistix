import abc
from typing import Any, Union

import equinox as eqx
from jaxtyping import ArrayLike, PyTree, Scalar


_Out = Union[Scalar, PyTree[ArrayLike]]
_ConstraintOut = Union[
    tuple[None, PyTree[ArrayLike]],
    tuple[PyTree[ArrayLike], None],
    tuple[PyTree[ArrayLike], PyTree[ArrayLike]],
]


class AbstractProblem(eqx.Module, strict=True):
    """Abstract base class for benchmark problems."""

    def name(self):
        """Returns the name of the benchmark problem, which should be the same as the
        name of the class that implements it. For CUTEST problems, this is the name of
        the problem used in the SIF file: e.g. "BT1" or "AIRCRAFTB".
        """
        return self.__class__.__name__

    @abc.abstractmethod
    def objective(self, y, args) -> _Out:
        """Objective function to be minimized. Can return a single scalar value (for a
        minimisation problem) or a PyTree of arrays (for a least-squares problem).
        """

    @abc.abstractmethod
    def y0(self) -> PyTree[ArrayLike]:
        """Initial guess for the optimization problem."""

    @abc.abstractmethod
    def args(self) -> PyTree[Any]:
        """Additional arguments for the objective function."""

    @abc.abstractmethod
    def expected_result(self) -> PyTree[ArrayLike]:
        """Expected result of the optimization problem. Should be a PyTree of arrays
        with the same structure as `y0`."""

    @abc.abstractmethod
    def expected_objective_value(self) -> _Out:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        For a least-squares problem, this is a PyTree of residuals.
        """


class AbstractUnconstrainedMinimisation(AbstractProblem, strict=True):
    """Abstract base class for unconstrained minimisation problems. The objective
    function for these problems returns a single scalar value, and they have neither
    bounds on the variable `y` nor any other constraints.
    """

    @abc.abstractmethod
    def objective(self, y, args) -> Scalar:
        """Objective function to be minimized. Must return a scalar value."""

    @abc.abstractmethod
    def expected_objective_value(self) -> Scalar:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        """


class AbstractBoundedMinimisation(AbstractProblem, strict=True):
    """Abstract base class for bounded minimisation problems. The objective
    function for these problems returns a single scalar value, they specify bounds on
    the variable `y` but no other constraints.
    """

    @abc.abstractmethod
    def objective(self, y, args) -> Scalar:
        """Objective function to be minimized. Must return a scalar value."""

    @abc.abstractmethod
    def expected_objective_value(self) -> Scalar:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        """

    @abc.abstractmethod
    def bounds(self) -> PyTree[ArrayLike]:
        """Returns the bounds on the variable `y`. Should be a tuple (`lower`, `upper`)
        where `lower` and `upper` are PyTrees of arrays with the same structure as `y0`.
        """


class AbstractConstrainedMinimisation(AbstractProblem, strict=True):
    """Abstract base class for constrained minimisation problems. These can have both
    equality or inequality constraints, and they may also have bounds on `y`. We do not
    differentiate between bounded constrained problems and constrained optimisation
    problems without bounds, as we do expect our solvers to do the right thing in each
    of these cases.
    """

    @abc.abstractmethod
    def objective(self, y, args) -> Scalar:
        """Objective function to be minimized. Must return a scalar value."""

    @abc.abstractmethod
    def expected_objective_value(self) -> Scalar:
        """Expected value of the objective function at the optimal solution. For a
        minimisation function, this is a scalar value.
        """

    @abc.abstractmethod
    def bounds(self) -> Union[PyTree[ArrayLike], None]:
        """Returns the bounds on the variable `y`, if specified.
        Should be a tuple (`lower`, `upper`) where `lower` and `upper` are PyTrees of
        arrays with the same structure as `y0`.
        """

    @abc.abstractmethod
    def constraint(self, y) -> _ConstraintOut:
        """Returns the constraints on the variable `y`. The constraints can be either
        equality, inequality constraints, or both. This method returns a tuple, with the
        equality constraint in the first argument and the inequality constraint values
        in the second argument. If there are no equality constraints, the first element
        should be `None`. If there are no inequality constraints, the second element
        should be `None`. (None, None) is not allowed as an output - in that case the
        problem has no constraints and should not be classified as a constrained
        minimisation problem.

        All constraints are assumed to be satisfied when the value is
        equal to zero for equality constraints and greater than or equal to zero for
        inequality constraints. Each element of each returned pytree of arrays will be
        treated as the output of a constraint function (in other words: each constraint
        function returns a scalar value, a collection of which may be arranged in a
        pytree.)

        Example:
        ```python
        def constraint(self, y):
            x1, x2, x3 = y
            # Equality constraints
            c1 = x1 * x2 + x3
            # Inequality constraints
            c2 = x1 + x2
            c3 = x3 - x3
            return c1, (c2, c3)
        ```
        """
