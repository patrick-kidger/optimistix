import abc
import warnings
from collections.abc import Callable
from typing import Any, Generic, TYPE_CHECKING, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar
from jaxtyping import Array, Bool, PyTree, Scalar

from ._adjoint import AbstractAdjoint
from ._custom_types import (
    Aux,
    Constraint,
    EqualityOut,
    Fn,
    InequalityOut,
    Out,
    SolverState,
    Y,
)
from ._misc import unwrap_jaxpr, wrap_jaxpr
from ._search import Iterate
from ._solution import RESULTS, Solution


if TYPE_CHECKING:
    _Node = Any
else:
    _Node = eqxi.doc_repr(Any, "Node")


_Iterate = TypeVar("_Iterate", contravariant=True, bound=Iterate)


class AbstractIterativeSolver(
    eqx.Module, Generic[Y, _Iterate, Out, Aux, SolverState], strict=True
):
    """Abstract base class for all iterative solvers."""

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]

    @abc.abstractmethod
    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> tuple[_Iterate, SolverState]:
        """Perform all initial computation needed to initialise the solver state.

        For example, the [`optimistix.Chord`][] method computes the Jacobian `df/dy`
        with respect to the initial guess `y`, and then uses it throughout the
        computation.

        **Arguments:**

        - `fn`: The function to iterate over. This is expected to take two argumetns
            `fn(y, args)` and return a pytree of arrays in the first element, and any
            auxiliary data in the second argument.
        - `y`: The value of `y` at the current (first) iteration.
        - `args`: Passed as the `args` of `fn(y, args)`.
        - `options`: Individual solvers may accept additional runtime arguments.
            See each individual solver's documentation for more details.
        - `constraint`: Individual solvers may accept a constraint function
            `constraint(y)`. This must return a PyTree of scalars, one for each
            constraint, such that `constraint(y) >= 0` if the constraint is satisfied.
            Keyword only argument.
        - `bounds`: Individual solvers may accept bounds. This should be a pair of
            pytrees of the same structure as `y`, where the first element is the lower
            bound and the second is the upper bound. Unbounded leaves can be indicated
            with +/- inf for the upper and lower bounds, respectively (float('inf'),
            np.inf or jnp.inf, as needed to match the data types in `y`). For finite
            bounds, it is checked whether `y` is in the closed interval`[lower, upper]`.
        - `f_struct`: A pytree of `jax.ShapeDtypeStruct`s of the same shape as the
            output of `fn`. This is used to initialise any information in the state
            which may rely on the pytree structure, array shapes, or dtype of the
            output of `fn`.
        - `aux_struct`: A pytree of `jax.ShapeDtypeStruct`s of the same shape as the
            auxiliary data returned by `fn`.
        - `tags`: exact meaning depends on whether this is a fixed point, root find,
            least squares, or minimisation problem; see their relevant entry points.

        **Returns:**

        A PyTree representing the initial state of the solver.
        """

    @abc.abstractmethod
    def step(
        self,
        fn: Fn[Y, Out, Aux],
        iterate: _Iterate,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        state: SolverState,
        tags: frozenset[object],
    ) -> tuple[_Iterate, SolverState, Aux]:
        """Perform one step of the iterative solve.

        **Arguments:**

        - `fn`: The function to iterate over. This is expected to take two argumetns
            `fn(y, args)` and return a pytree of arrays in the first element, and any
            auxiliary data in the second argument.
        - `iterate`: The value of the iterate at the current (first) iteration. For an
            unconstrained problem, this is just `y`, wrapped in
            [`optimistix.Iterate.Primal`][]. For a constrained problem, the iterate
            comprises the primal and dual variables, as well as potential slack
            variables and other auxiliary optimisation variables, such as adaptively
            updated barrier parameters.
        - `args`: Passed as the `args` of `fn(y, args)`.
        - `options`: Individual solvers may accept additional runtime arguments.
            See each individual solver's documentation for more details.
        - `constraint`: Individual solvers may accept a constraint function
            `constraint(y)`. This must return a PyTree of scalars, one for each
            constraint, such that `constraint(y) >= 0` if the constraint is satisfied.
        - `bounds`: Individual solvers may accept bounds. This should be a pair of
            pytrees of the same structure as `y`, where the first element is the lower
            bound and the second is the upper bound. Unbounded leaves can be indicated
            with +/- inf for the upper and lower bounds, respectively (float('inf'),
            np.inf or jnp.inf, as needed to match the data types in `y`). For finite
            bounds, it is checked whether `y` is in the closed interval`[lower, upper]`.
        - `state`: A pytree representing the state of a solver. The shape of this
            pytree is solver-dependent.
        - `tags`: exact meaning depends on whether this is a fixed point, root find,
            least squares, or minimisation problem; see their relevant entry points.

        **Returns:**

        A 3-tuple containing the new `y` value in the first element, the next solver
        state in the second element, and the aux output of `fn(y, args)` in the third
        element.
        """

    # TODO update documentation
    @abc.abstractmethod
    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        iterate: _Iterate,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        state: SolverState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        """Determine whether or not to stop the iterative solve.

        **Arguments:**

        - `fn`: The function to iterate over. This is expected to take two argumetns
            `fn(y, args)` and return a pytree of arrays in the first element, and any
            auxiliary data in the second argument.
        - `y`: The value of `y` at the current iteration.
        - `args`: Passed as the `args` of `fn(y, args)`.
        - `options`: Individual solvers may accept additional runtime arguments.
            See each individual solver's documentation for more details.
        - `constraint`: Individual solvers may accept a constraint function
            `constraint(y)`. This must return a PyTree of scalars, one for each
            constraint, such that `constraint(y) >= 0` if the constraint is satisfied.
        - `bounds`: Individual solvers may accept bounds. This should be a pair of
            pytrees of the same structure as `y`, where the first element is the lower
            bound and the second is the upper bound. Unbounded leaves can be indicated
            with +/- inf for the upper and lower bounds, respectively (float('inf'),
            np.inf or jnp.inf, as needed to match the data types in `y`). For finite
            bounds, it is checked whether `y` is in the closed interval`[lower, upper]`.
        - `state`: A pytree representing the state of a solver. The shape of this
            pytree is solver-dependent.
        - `tags`: exact meaning depends on whether this is a fixed point, root find,
            least squares, or minimisation problem; see their relevant entry points.

        **Returns:**

        A 2-tuple containing a bool indicating whether or not to stop iterating in the
        first element, and an [`optimistix.RESULTS`][] object in the second element.
        """

    # TODO update documentation
    @abc.abstractmethod
    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        iterate: _Iterate,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
        bounds: tuple[Y, Y] | None,
        state: SolverState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        """Any final postprocessing to perform on the result of the solve.

        **Arguments:**

        - `fn`: The function to iterate over. This is expected to take two argumetns
            `fn(y, args)` and return a pytree of arrays in the first element, and any
            auxiliary data in the second argument.
        - `y`: The value of `y` at the last iteration.
        - `aux`: The auxiliary output at the last iteration.
        - `args`: Passed as the `args` of `fn(y, args)`.
        - `options`: Individual solvers may accept additional runtime arguments.
            See each individual solver's documentation for more details.
        - `constraint`: Individual solvers may accept a constraint function
            `constraint(y)`. This must return a PyTree of scalars, one for each
            constraint, such that `constraint(y) >= 0` if the constraint is satisfied.
            Keyword only argument.
        - `bounds`: Individual solvers may accept bounds. This should be a pair of
            pytrees of the same structure as `y`, where the first element is the lower
            bound and the second is the upper bound. Unbounded leaves can be indicated
            with +/- inf for the upper and lower bounds, respectively (float('inf'),
            np.inf or jnp.inf, as needed to match the data types in `y`). For finite
            bounds, it is checked whether `y` is in the closed interval`[lower, upper]`.
        - `state`: A pytree representing the final state of a solver. The shape of this
            pytree is solver-dependent.
        - `tags`: exact meaning depends on whether this is a fixed point, root find,
            least squares, or minimisation problem; see their relevant entry points.
        - `result`: as returned by the final call to `terminate`.

        **Returns:**

        A 3-tuple of:

        - `final_y`: the final `y` to return as the solution of the solve.
        - `final_aux`: the final `aux` to return as the auxiliary output of the solve.
        - `stats`: any additional information to place in the `sol.stats` dictionary.

        !!! info

            Most solvers will not need to use this, so that this method may be defined
            as:
            ```python
            def postprocess(self, fn, y, aux, args, options, state, tags, result):
                return y, aux, {}
            ```
        """


def _zero(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jnp.zeros(x.shape, dtype=x.dtype)
    else:
        return x


def _iterate(inputs):
    (
        fn,
        solver,
        y0,
        args,
        options,
        constraint,
        bounds,
        max_steps,
        f_struct,
        aux_struct,
        tags,
        while_loop,
    ) = inputs
    del inputs
    static_leaf = lambda x: isinstance(x, eqxi.Static)
    f_struct = jtu.tree_map(lambda x: x.value, f_struct, is_leaf=static_leaf)
    aux_struct = jtu.tree_map(lambda x: x.value, aux_struct, is_leaf=static_leaf)
    init_aux = jtu.tree_map(_zero, aux_struct)
    init_iterate, init_state = solver.init(
        fn, y0, args, options, constraint, bounds, f_struct, aux_struct, tags
    )
    dynamic_init_state, static_state = eqx.partition(init_state, eqx.is_array)
    init_carry = (
        init_iterate,
        jnp.array(0),
        dynamic_init_state,
        init_aux,
    )

    def cond_fun(carry):
        iterate, _, dynamic_state, _ = carry
        state = eqx.combine(static_state, dynamic_state)
        terminate, result = solver.terminate(
            fn, iterate, args, options, constraint, bounds, state, tags
        )
        return jnp.invert(terminate) | (result != RESULTS.successful)

    def body_fun(carry):
        iterate, num_steps, dynamic_state, _ = carry
        state = eqx.combine(static_state, dynamic_state)
        new_iterate, new_state, aux = solver.step(
            fn, iterate, args, options, constraint, bounds, state, tags
        )
        new_dynamic_state, new_static_state = eqx.partition(new_state, eqx.is_array)

        assert eqx.tree_equal(static_state, new_static_state) is True
        return new_iterate, num_steps + 1, new_dynamic_state, aux

    final_carry = while_loop(cond_fun, body_fun, init_carry, max_steps=max_steps)
    final_iterate, num_steps, dynamic_final_state, final_aux = final_carry
    final_state = eqx.combine(static_state, dynamic_final_state)
    terminate, result = solver.terminate(
        fn, final_iterate, args, options, constraint, bounds, final_state, tags
    )
    result = RESULTS.where(
        (result == RESULTS.successful) & jnp.invert(terminate),
        RESULTS.nonlinear_max_steps_reached,
        result,
    )
    final_y, final_aux, stats = solver.postprocess(
        fn,
        final_iterate,
        final_aux,
        args,
        options,
        constraint,
        bounds,
        final_state,
        tags,
        result,
    )
    return final_y, (
        num_steps,
        result,
        dynamic_final_state,
        eqxi.Static(wrap_jaxpr(static_state)),
        final_aux,
        stats,
    )


# Keep `optx.implicit_jvp` is happy.
if _iterate.__globals__["__name__"].startswith("jaxtyping"):
    _iterate = _iterate.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


def iterative_solve(
    fn: Fn[Y, Out, Aux],
    # no type parameters, see https://github.com/microsoft/pyright/discussions/5599
    solver: AbstractIterativeSolver,
    y0: PyTree[Array],
    args: PyTree = None,
    options: dict[str, Any] | None = None,
    *,
    constraint: Constraint[Y, EqualityOut, InequalityOut] | None,
    bounds: tuple[Y, Y] | None,
    max_steps: int | None,
    adjoint: AbstractAdjoint,
    throw: bool,
    tags: frozenset[object],
    f_struct: PyTree[jax.ShapeDtypeStruct],
    aux_struct: PyTree[jax.ShapeDtypeStruct],
    rewrite_fn: Callable,
) -> Solution[Y, Aux]:
    """Compute the iterates of an iterative numerical method.

    Given a nonlinear function `fn(y, args)` and an iterative method `solver`,
    this computes the iterates generated by `solver`. This generalises minimisation,
    least-squares, root-finding, and fixed-point iteration to any iterative
    numerical method applied to `fn(y, args)`.

    **Arguments:**

    - `fn`: The function to iterate over. This is expected to take two arguments
        `fn(y, args)` and return a pytree of arrays in the first element, and any
        auxiliary data in the second argument.
    - `solver`: The solver to use. This should be a subclass of
        [`optimistix.AbstractIterativeSolver`][].
    - `y0`: An initial guess for what `y` may be.
    - `args`: Passed as the `args` of `fn(y, args)`.
    - `options`: Individual solvers may accept additional runtime arguments.
        See each individual solver's documentation for more details.
    - `constraint`: Individual solvers may accept a constraint function `constraint(y)`.
        This must return a PyTree of scalars, one for each constraint, such that
        `constraint(y) >= 0` if the constraint is satisfied. Keyword only argument.
    - `bounds`: Individual solvers may accept bounds. This should be a pair of pytrees
        of the same structure as `y`, where the first element is the lower bound, and
        the second is the upper bound. Unbounded leaves can be indicated with +/- inf
        for the upper and lower bounds, respectively (float('inf'), np.inf or jnp.inf,
        as needed to match the data types in `y`). For finite bounds, it is checked
        whether `y` is in the closed interval`[lower, upper]`. Keyword only argument.
    - `max_steps`: The maximum number of steps the solver can take. Keyword only
        argument.
    - `adjoint`: The adjoint method used to compute gradients through an iterative
        solve. Keyword only argument.
    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or encountering divergent iterates.) If `True` then a failure will
        raise an error. If `False` then the returned solution object will have a
        `result` field indicating whether any failures occured. (See
        [`optimistix.Solution`][].) Keyword only argument.
    - `tags`: exact meaning depends on whether this is a fixed point, root find,
        least squares, or minimisation problem; see their relevant entry points.
    - `f_struct`: A pytree of `jax.ShapeDtypeStruct`s of the same shape as the
        output of `fn`. This is used to initialise any information in the state
        which may rely on the pytree structure, array shapes, or dtype of the
        output of `fn`. Keyword only argument.
    - `aux_struct`: A pytree of `jax.ShapeDtypeStruct`s of the same shape as the
        auxiliary data returned by `fn`. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """
    if options is None:
        options = {}

    if any(jnp.iscomplexobj(x) for x in jtu.tree_leaves((y0, f_struct))):
        warnings.warn(
            "Complex support in Optimistix is a work in progress, and may still return "
            "incorrect results. You may prefer to split your problem into real and "
            "imaginary parts, so that Optimistix sees only real numbers."
        )

    f_struct = jtu.tree_map(eqxi.Static, f_struct)
    aux_struct = jtu.tree_map(eqxi.Static, aux_struct)
    inputs = (
        fn,
        solver,
        y0,
        args,
        options,
        constraint,
        bounds,
        max_steps,
        f_struct,
        aux_struct,
        tags,
    )
    (
        out,
        (
            num_steps,
            result,
            dynamic_final_state,
            static_state,
            aux,
            stats,
        ),
    ) = adjoint.apply(_iterate, rewrite_fn, inputs, tags)
    final_state = eqx.combine(dynamic_final_state, unwrap_jaxpr(static_state.value))
    stats = {"num_steps": num_steps, "max_steps": max_steps, **stats}
    sol = Solution(value=out, result=result, state=final_state, aux=aux, stats=stats)
    if throw:
        sol = result.error_if(sol, result != RESULTS.successful)
    return sol
