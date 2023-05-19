from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Float, Int, PyTree

from ..line_search import OneDimensionalFunction
from ..linear_operator import AbstractLinearOperator
from ..minimise import AbstractMinimiser, MinimiseProblem
from ..misc import tree_inner_prod
from ..solution import RESULTS
from .misc import get_vector_operator


class BacktrackingState(eqx.Module):
    f_delta: Float[Array, ""]
    f0: Float[Array, ""]
    vector: PyTree[Array]
    operator: AbstractLinearOperator
    diff: PyTree[Array]
    compute_f0: Bool[Array, ""]
    result: RESULTS
    step: Int[Array, ""]


class BacktrackingArmijo(AbstractMinimiser):
    backtrack_slope: float
    decrease_factor: float
    gauss_newton: bool

    def search_init(
        self,
        problem: MinimiseProblem[OneDimensionalFunction],
        y: Array,
        args: Any,
        options: dict[str, Any],
    ):
        return 1.0

    def init(
        self,
        problem: MinimiseProblem[OneDimensionalFunction],
        y: Array,
        args: Any,
        options: dict[str, Any],
    ):
        f0, (_, diff, *_) = jtu.tree_map(
            lambda x: jnp.zeros(shape=x.shape), jax.eval_shape(problem.fn, 0.0, None)
        )
        vector, operator = get_vector_operator(options)

        try:
            f0 = options["f0"]
            compute_f0 = options["compute_f0"]
        except KeyError:
            compute_f0 = jnp.array(True)

        return BacktrackingState(
            f_delta=f0,
            f0=f0,
            diff=diff,
            vector=vector,
            operator=operator,
            compute_f0=compute_f0,
            result=jnp.array(RESULTS.successful),
            step=jnp.array(0),
        )

    def step(
        self,
        problem: MinimiseProblem[OneDimensionalFunction],
        y: Array,
        args: Any,
        options: dict[str, Any],
        state: BacktrackingState,
    ):
        _delta = y * self.decrease_factor
        delta = jnp.where(state.compute_f0, jnp.array(0.0), _delta)
        (f_delta, (_, diff, aux, result)) = problem.fn(delta, args)
        f0 = jnp.where(state.compute_f0, f_delta, state.f0)
        new_state = BacktrackingState(
            f_delta,
            f0,
            diff,
            state.vector,
            state.operator,
            jnp.array(False),
            result,
            state.step + 1,
        )
        return _delta, new_state, (f_delta, diff, aux, result, jnp.array(1.0))

    def terminate(
        self,
        problem: MinimiseProblem[OneDimensionalFunction],
        y: Array,
        args: Any,
        options: dict[str, Any],
        state: BacktrackingState,
    ):
        result = jnp.where(
            jnp.isfinite(y),
            state.result,
            RESULTS.nonlinear_divergence,  # pyright: ignore
        )
        if self.gauss_newton:
            grad = state.operator.transpose().mv(state.vector)
        else:
            grad = state.vector
        predicted_reduction = tree_inner_prod(grad, state.diff)
        # WARNING: this is a foot gun
        predicted_reduction = jnp.minimum(predicted_reduction, 0)
        finished = state.f_delta < state.f0 + self.backtrack_slope * predicted_reduction
        finished = finished & (state.step > 1)
        return finished, result

    def buffers(self, state: BacktrackingState):
        return ()
