from collections.abc import Callable
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._fixed_point import AbstractFixedPointSolver
from .._misc import max_norm
from .._solution import RESULTS


class _AndersonAccelFixedPointState(eqx.Module, Generic[Y]):
    relative_error: Scalar
    past_errors: Y
    past_iterates: Y
    right_hand_side: tuple[Scalar, Array]
    current_iter: Int[Array, ""]


class AndersonAcceleration(
    AbstractFixedPointSolver[Y, Aux, _AndersonAccelFixedPointState]
):
    r"""Repeatedly calls a function in search of a fixed point.

    Unlike the usual fixed point iterations that simply
    repeatedly call `y_{n+1}=f(y_n)` until `y_n` stops changing,
    Anderson methods attempt to blend residuals (`f(y_k)-y_k`) and
    iterates (`y_k`) from a window of past iterations to produce
    the next iterate.

    The update formula amounts to finding `α` solving the following
    optimisation problem.

    ```
    min || G.α ||^2
    s.t. 1^T.α = 1
         G = [*residuals_past]
    ```
    The KKT conditions of this problem yield a system of linear equations to
    solve at each iteration, of size `dim(y)+1`.
    ```
    [ 0  1^T  ] [ν]   [1]
    [         ] [ ]   [ ]
    [ 1 G^T.G ] [α] = [0]
    [         ] [ ]   [ ]
    ```
    Then combine the past residuals and iterates:
    ```
    new_y = β * dot(G, α) + dot([*y_past], α)
    ```

    See e.g.
        - [this tutorial](https://implicit-layers-tutorial.org/implicit_functions/).
        - [[Fang and Saad, 2009]](https://onlinelibrary.wiley.com/doi/10.1002/nla.617)
        - [[Walker and Ni, 2011]](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf)
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    # Beta coefficient for damping
    mixing: float = 0.5
    memory_size: int = 10
    # Regularization factor to lift up the smallest e.v. of the normal equations:
    # Cf. doi.org/10.1016/j.anucene.2024.110837 eq (42)->(43)
    regularization: float = 1e-6
    linear_solver: lx.AbstractLinearSolver = lx.LU()

    def init(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _AndersonAccelFixedPointState:
        del options, f_struct, aux_struct, tags
        fn_output, _ = fn(y, args)

        with jax.numpy_dtype_promotion("standard"):
            # The matrix involved in the normal equations (G)
            error_memory_init = jax.tree.map(
                lambda _f, _y: jnp.stack([_f - _y] * self.memory_size, axis=0),
                fn_output,
                y,
            )
            # [z_k, z_{k-1}, ...]
            iterates_memory_init = jax.tree.map(
                lambda _y: jnp.stack([_y] * self.memory_size, axis=0), y
            )
            # Get the kind of (nu, alpha) combination wished for, by computing
            # the dtype of the output. _infos contains the dtype of a scalar that
            # corresponds to the sum of all the coordinates (in pytree form) of
            # the function's output
            _infos = jax.eval_shape(
                lambda _y: jax.tree.reduce(jnp.add, (_y**ω).call(jnp.sum).ω), fn_output
            )
        # Now that we used context management to get the "common dtype" of the
        # outputs of the function, i.e. of the linear system to solve, we can
        # build the right hand side of the normal equations
        right_hand_side = (
            jnp.ones_like(_infos),
            jnp.zeros((self.memory_size,)).astype(_infos.dtype),
        )
        return _AndersonAccelFixedPointState(
            jnp.array(jnp.inf),
            error_memory_init,
            iterates_memory_init,
            right_hand_side,
            jnp.array(1),
        )

    def step(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonAccelFixedPointState,
        tags: frozenset[object],
    ) -> tuple[Y, _AndersonAccelFixedPointState, Aux]:
        del tags
        fn_output, aux = fn(y, args)
        error = y**ω - fn_output**ω

        # Update the rolling memory without shifting the data
        new_errors_memory: Y = jax.tree.map(
            lambda olds_, new_: olds_.at[state.current_iter % self.memory_size].set(
                new_
            ),
            state.past_errors,
            error.ω,
        )
        new_iterates_memory: Y = jax.tree.map(
            lambda olds_, new_: olds_.at[state.current_iter % self.memory_size].set(
                new_
            ),
            state.past_iterates,
            fn_output,
        )
        with jax.numpy_dtype_promotion("standard"):

            def kkt_linear_op(
                optim_vector: tuple[Scalar, Array],
            ) -> tuple[Scalar, Array]:
                """
                Solve:
                [ 0 1...1 ] ( ν )   ( 1 )
                | 1       | |   |   | 0 |
                | : G^T.G | | α | = | : |
                [ 1       ] (   )   ( 0 )
                This function describes the left operator.
                """
                nu, alpha = optim_vector
                normal_equation = nu + jax.tree.reduce(
                    lambda _carry, _errs: _carry
                    + jnp.einsum("j...,i...,i->j", _errs, _errs, alpha),
                    new_errors_memory,
                    initializer=jnp.array(0.0),
                )
                return (jnp.sum(alpha), normal_equation + self.regularization * alpha)

            # Pointer to the right-hand side of the equations
            rhs = state.right_hand_side

            solution = lx.linear_solve(
                # In case the user wishes to use GMRES/NormalCG as solver,
                # we define the linear operator lazily without materializing the
                # matrix
                lx.FunctionLinearOperator(
                    kkt_linear_op,
                    jax.eval_shape(lambda: rhs),
                    # The normal equations matrix is always at least symmetric.
                    # It has one negative eigenvalue, and n positive ones.
                    tags=frozenset({lx.symmetric_tag}),
                ),
                rhs,
                self.linear_solver,
                throw=False,
                options=options,
            )
            _, weighting = solution.value

            # New iterate is G.α
            new_y = jax.tree.map(
                eqx.Partial(self._combine_iters, weighting),
                new_errors_memory,
                new_iterates_memory,
            )

            scale = self.atol + self.rtol * ω(new_y).call(jnp.abs)
            new_state = _AndersonAccelFixedPointState(
                self.norm((error / scale).ω),
                new_errors_memory,
                new_iterates_memory,
                state.right_hand_side,
                state.current_iter + 1,
            )
        return new_y, new_state, aux

    def _combine_iters(
        self, weighting: Array, errors: PyTree[Array], iters: PyTree[Array]
    ):
        # Helper function to combine past iterates and past residuals into next
        # iterates.
        return jnp.einsum("i...,i->...", iters + self.mixing * errors, weighting)

    def terminate(
        self,
        fn: Fn[Y, Y, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonAccelFixedPointState,
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
        state: _AndersonAccelFixedPointState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


AndersonAcceleration.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `mixing`: Beta coefficient for damping. Defaults to `1.`
- `memory_size`: Number of vectors to keep in memory, both for past residuals
    and past iterates. Defaults to `10`
- `regularization: regularization factor to lift up the smallest e.v. of the
    normal equations. Defaults to `1e-6`
- `linear_solver: Solver from the [`lineax`][] library. Defaults to [`lx.LU()`][]
"""
