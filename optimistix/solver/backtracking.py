from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, Int, PyTree

from ..line_search import OneDimensionalFunction
from ..minimise import AbstractMinimiser, MinimiseProblem
from ..misc import tree_full, tree_inner_prod, tree_where, tree_zeros_like
from ..solution import RESULTS
from .misc import get_vector_operator


class BacktrackingState(eqx.Module):
    f_delta: Float[Array, ""]
    f0: Float[Array, ""]
    running_min: Array
    running_min_diff: PyTree[Array]
    vector: PyTree[Array]
    operator: lx.AbstractLinearOperator
    diff: PyTree[Array]
    compute_f0: Bool[Array, ""]
    result: RESULTS
    step: Int[Array, ""]


class BacktrackingArmijo(AbstractMinimiser):
    gauss_newton: bool
    backtrack_slope: float
    decrease_factor: float

    def first_init(self, vector, operator, options):
        try:
            init_size = options["init_line_search"]
        except KeyError:
            if jax.config.jax_enable_x64:
                eps = jnp.finfo(jnp.float64).eps
            else:
                eps = jnp.finfo(jnp.float64).eps
            if self.decrease_factor < eps:
                raise ValueError("The decrease factor of linesearch must be nonzero!")
            init_size = 1 / self.decrease_factor
        return init_size
        ...

    def init(
        self,
        problem: MinimiseProblem[OneDimensionalFunction],
        y: Array,
        args: Any,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
    ):
        f0 = tree_full(f_struct, jnp.inf)
        vector, operator = get_vector_operator(options)
        try:
            diff0 = tree_zeros_like(options["diff"])
        except KeyError:
            assert False

        try:
            f0 = options["f0"]
            compute_f0 = options["compute_f0"]
        except KeyError:
            compute_f0 = jnp.array(True)

        return BacktrackingState(
            f_delta=f0,
            f0=f0,
            running_min=f0,
            running_min_diff=diff0,
            vector=vector,
            operator=operator,
            diff=diff0,
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
        delta = jnp.where(state.compute_f0, jnp.array(0.0), y)
        (f_delta, (_, diff, aux, result, _)) = problem.fn(delta, args)
        f0 = jnp.where(state.compute_f0, f_delta, state.f0)
        running_min = jnp.where(f_delta < state.running_min, f_delta, state.running_min)
        running_min_diff = tree_where(
            f_delta < state.running_min, diff, state.running_min_diff
        )
        new_state = BacktrackingState(
            f_delta,
            f0,
            running_min,
            running_min_diff,
            state.vector,
            state.operator,
            diff,
            jnp.array(False),
            result,
            state.step + 1,
        )
        new_y = self.decrease_factor * y
        return (
            new_y,
            new_state,
            (running_min, running_min_diff, aux, result, jnp.array(1.0)),
        )

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
        predicted_reduction = jnp.minimum(predicted_reduction, 0)
        finished = state.f_delta < state.f0 + self.backtrack_slope * predicted_reduction
        finished = finished & (state.step > 1)
        return finished, result

    def buffers(self, state: BacktrackingState):
        return ()
