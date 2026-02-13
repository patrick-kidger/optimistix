from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._fixed_point import AbstractFixedPointSolver
from .._misc import max_norm
from .._solution import RESULTS


class _FixedPointState(eqx.Module):
    relative_error: Scalar


class FixedPointIteration(AbstractFixedPointSolver[Y, Aux, _FixedPointState]):
    """Repeatedly calls a function in search of a fixed point.

    This is one of the simplest ways to find a fixed point `y` of `f`: simply
    repeatedly call `y_{n+1}=f(y_n)` until `y_n` stops changing.

    Optionally, one can use damping between iterations, then the update becomes
    `y_{n+1}=(1-damp)*f(y_n)+damp*y_n` where `damp` is the damping factor.

    Note that this is often not a very effective method, and root-finding algorithms are
    frequently preferred in practice.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    damp: float = 0.0

    def init(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _FixedPointState:
        del fn, y, args, options, f_struct, aux_struct
        return _FixedPointState(jnp.array(jnp.inf))

    def step(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Y, _FixedPointState, Aux]:
        fn_y_n, aux = fn(y, args)
        new_y = jtu.tree_map(
            lambda i_np1, i_n: (1 - self.damp) * i_np1 + self.damp * i_n, fn_y_n, y
        )
        error = (y**ω - new_y**ω).ω
        with jax.numpy_dtype_promotion("standard"):
            scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
            new_state = _FixedPointState(self.norm((error**ω / scale**ω).ω))
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.relative_error < 1, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _FixedPointState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


def _batched_tree_zeros_like(y, batch_dimension):
    return jtu.tree_map(lambda y: jnp.zeros((*y.shape, batch_dimension)), y)


def _convert_to_pytree_op(tree, output_structure, i, batch_dimension):
    """
    Take rolling buffer tree and compute diff between iterations, and create operator
    """

    indices = jnp.maximum(i - jnp.arange(0, batch_dimension - 1), 0) % batch_dimension
    shift_indices = jnp.maximum(i - jnp.arange(1, batch_dimension), 0) % batch_dimension
    diff_tree = jtu.tree_map(
        lambda leaf, shift_leaf: leaf[..., indices] - shift_leaf[..., shift_indices],
        tree,
        tree,
    )

    pytree_op = lx.PyTreeLinearOperator(diff_tree, output_structure)
    return pytree_op


class _AndersonAccelerationState(eqx.Module, Generic[Y]):
    first_step: Bool[Array, ""]
    index_start: Scalar
    history_length: int
    f_history: PyTree[Y]
    g_history: PyTree[Y]
    output_structure: PyTree
    relative_error: Scalar


class AndersonAcceleration(
    AbstractFixedPointSolver[Y, Aux, _AndersonAccelerationState]
):
    """
    Solves the fixed-point equation

        x = f(x)

    by applying Anderson acceleration to the fixed-point iteration.

    Let

        g_k = f(x_k) - x_k

    denote the residual at iteration k. Using the last `m` residual and
    iterate differences (with `m = history_length`), Anderson acceleration
    computes coefficients γ_k by solving the least-squares problem

        γ_k = argmin_γ || g_k - Δg_k γ ||_2,

    where Δg_k contains differences of recent residuals.

    The next iterate is then computed as

        x_{k+1} = f(x_k) - Δf_k γ_k,

    where Δf_k contains differences of recent mapped iterates f(x).

    A damping factor `damp ∈ [0, 1]` optionally blends the accelerated
    update with the previous iterate:

        x_{k+1} ← (1 - damp) * x_{k+1} + damp * x_k.

    Reference:
    D. G. Anderson. Iterative procedures for nonlinear integral equations. J. Assoc.
    Comput. Machinery, 12:547–560, 1965.
    """

    rtol: float
    atol: float
    history_length: int = 10
    norm: Callable[[PyTree], Scalar] = max_norm
    damp: float = 0.0

    def init(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _AndersonAccelerationState:
        del fn, args, options, f_struct, aux_struct
        state = _AndersonAccelerationState(
            first_step=jnp.array(True),
            index_start=jnp.array(0),
            history_length=self.history_length,
            f_history=_batched_tree_zeros_like(y, self.history_length),
            g_history=_batched_tree_zeros_like(y, self.history_length),
            output_structure=jax.eval_shape(lambda: y),
            relative_error=jnp.array(jnp.inf),
        )
        return state

    def step(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonAccelerationState,
        tags: frozenset[object],
    ) -> tuple[Y, _AndersonAccelerationState, Aux]:
        fn_y_n, aux = fn(y, args)
        g_y_n = jtu.tree_map(lambda a, b: a - b, fn_y_n, y)

        # Add new results to history
        f_history = jtu.tree_map(
            lambda x, z: x.at[..., state.index_start % self.history_length].set(z),
            state.f_history,
            fn_y_n,
        )

        g_history = jtu.tree_map(
            lambda x, z: x.at[..., state.index_start % self.history_length].set(z),
            state.g_history,
            g_y_n,
        )

        # Perform Anderson acceleration
        # First iterate must be treated differently
        def _first_iterate():
            new_y = jtu.tree_map(
                lambda y_old, y_next: self.damp * y_old + (1 - self.damp) * y_next,
                y,
                fn_y_n,
            )
            return new_y

        def _find_new_y():
            # Construct F operator
            F_op = _convert_to_pytree_op(
                f_history,
                state.output_structure,
                state.index_start,
                self.history_length,
            )
            # Construct G operator
            G_op = _convert_to_pytree_op(
                g_history,
                state.output_structure,
                state.index_start,
                self.history_length,
            )

            # Compute gammas
            # Use SVD - for initial iterations, F has columns of zeros
            gammas = lx.linear_solve(G_op, g_y_n, solver=lx.SVD()).value

            # Compute terms for update and include damping
            F_gamma = F_op.mv(gammas)

            y_undamped = jtu.tree_map(
                lambda f_n, F_gamma: f_n - F_gamma, fn_y_n, F_gamma
            )

            new_y = jtu.tree_map(
                lambda y_undamped, y_damped_update: (1 - self.damp) * y_undamped
                + self.damp * y_damped_update,
                y_undamped,
                y,
            )
            return new_y

        new_y = jax.lax.cond(state.first_step, _first_iterate, _find_new_y)

        error = (y**ω - new_y**ω).ω
        with jax.numpy_dtype_promotion("standard"):
            scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
            rel_err = self.norm((error**ω / scale**ω).ω)

        new_state = _AndersonAccelerationState(
            first_step=jnp.array(False),
            index_start=state.index_start + 1,
            history_length=self.history_length,
            f_history=f_history,
            g_history=g_history,
            output_structure=state.output_structure,
            relative_error=rel_err,
        )

        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonAccelerationState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.relative_error < 1, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonAccelerationState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


FixedPointIteration.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `damp`: The damping factor used in iteration update.
"""

AndersonAcceleration.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `history_length`: Number of previous iterations used in residuals matrix.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `damp`: The damping factor used in iteration update.
"""
