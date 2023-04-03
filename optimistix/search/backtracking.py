from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax.lax as lax
from jaxtyping import Array, ArrayLike, Float, PyTree

from ..custom_types import sentinel
from ..iterate import AbstractIterativeProblem
from ..line_search import AbstractLineSearch, LineSearchState


class BacktrackingArmijo(AbstractLineSearch):
    def search(
        self,
        problem: AbstractIterativeProblem,
        y: PyTree,
        search_state: LineSearchState,
        args: PyTree,
        options: Dict[str, Any],
        *,
        f_y: Optional[PyTree[Array]] = sentinel,
        gradient: Optional[PyTree[Array]] = sentinel,
        Hessian: Optional[PyTree[Array]] = sentinel
    ) -> Tuple[Float[ArrayLike, " "], LineSearchState]:

        if f_y == sentinel:
            (f_y, aux) = problem.fn(y, args)
        if gradient == sentinel:
            (gradient, aux) = eqx.filter_grad(problem.fn, has_aux=problem.has_aux)(y)

        linear_decrease = gradient.T @ gradient

        # TODO(raderj): allow general delta_init
        # TODO(raderj): put in a general backtracking procedure and inherit/subclass
        delta = 2
        slope = 1

        def cond_fun(carry):
            f_new, delta = carry
            return f_new <= f_y + delta * slope * linear_decrease

        def body_fun(carry):
            f_new, delta = carry
            delta = delta / 2
            f_new = problem.fn(f_y + delta * gradient)
            return f_new, delta

        init = (f_y, delta)
        _, delta = lax.while_loop(cond_fun, body_fun, init)
        return (delta, search_state)
