from typing import Any, Generic, Union

import equinox as eqx
import jax
from equinox import AbstractVar
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import (
    Aux,
    Constraint,
    EqualityOut,
    Fn,
    InequalityOut,
    Out,
    SolverState,
    Y,
)
from .._iterate import AbstractIterativeSolver
from .._minimise import AbstractMinimiser
from .._solution import RESULTS
from .boundary_maps import AbstractBoundaryMap
from .gradient_methods import AbstractGradientDescent


class _ProjectedState(eqx.Module, Generic[SolverState], strict=True):
    wrapped_state: SolverState


class _AbstractProjectedSolver(
    AbstractIterativeSolver, Generic[Y, Out, Aux], strict=True
):
    solver: AbstractVar[AbstractIterativeSolver[Y, Out, Aux, Any]]
    boundary_map: AbstractVar[AbstractBoundaryMap]

    @property  # pyright: ignore
    def rtol(self):
        return self.solver.rtol

    @property  # pyright: ignore
    def atol(self):
        return self.solver.atol

    @property  # pyright: ignore
    def norm(self):
        return self.solver.norm

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _ProjectedState:
        state = self.solver.init(
            fn,
            y,
            args,
            options,
            constraint,
            bounds,
            f_struct,
            aux_struct,
            tags,
        )
        return _ProjectedState(wrapped_state=state)

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _ProjectedState,
        tags: frozenset[object],
    ) -> tuple[Y, _ProjectedState, Aux]:
        new_y, new_state, new_aux = self.solver.step(
            fn, y, args, options, constraint, bounds, state.wrapped_state, tags
        )
        new_y_eval, _ = self.boundary_map(new_state.y_eval, constraint, bounds)
        new_state = eqx.tree_at(lambda s: s.y_eval, new_state, new_y_eval)
        return new_y, _ProjectedState(wrapped_state=new_state), new_aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _ProjectedState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return self.solver.terminate(
            fn, y, args, options, constraint, bounds, state.wrapped_state, tags
        )

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _ProjectedState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


# TODO: As formulated here, these could be GradientDescent, or NonlinearCG. Do both of
# these make sense or should support be restricted to just GradientDescent? For now this
# is just tested for GradientDescent.
class ProjectedGradientDescent(  # pyright: ignore
    _AbstractProjectedSolver[Y, Scalar, Aux],
    AbstractMinimiser[Y, Aux, _ProjectedState],
    strict=True,
):
    """Projected gradient descent solver. Wraps a gradient descent solver and applies a
    boundary map to project every iterate onto the feasible set, if required.
    The boundary map should be an instance of [optimistix.AbstractBoundaryMap][].
    """

    solver: AbstractGradientDescent[Y, tuple[Scalar, Aux]]
    boundary_map: AbstractBoundaryMap

    # Explicitly declare to keep pyright happy.
    def __init__(
        self,
        solver: AbstractGradientDescent[Y, tuple[Scalar, Aux]],
        boundary_map: AbstractBoundaryMap,
    ):
        self.solver = solver
        self.boundary_map = boundary_map

    # Redeclare these three to work around the Equinox bug fixed here:
    # https://github.com/patrick-kidger/equinox/pull/544
    @property  # pyright: ignore
    def rtol(self):
        return self.solver.rtol

    @property  # pyright: ignore
    def atol(self):
        return self.solver.atol

    @property  # pyright: ignore
    def norm(self):
        return self.solver.norm


ProjectedGradientDescent.__init__.__doc__ = """**Arguments:**

- `solver`: the gradient descent solver to wrap.
- `boundary_map`: the boundary map to use.
"""


# TODO: Implement a proximal version that redefines fn using a proximal operator.
