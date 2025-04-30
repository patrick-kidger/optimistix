import abc
from typing import Any, Union

import equinox as eqx
from jaxtyping import ArrayLike, PyTree, Scalar


_Out = Union[Scalar, PyTree[ArrayLike]]


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
