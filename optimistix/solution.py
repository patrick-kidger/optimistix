from typing import Any, Dict, Union

import equinox as eqx
import equinox.internal as eqxi
from jaxtyping import Array, PyTree


_linear_singular_msg = """
The linear solver returned non-finite (NaN or inf) output. This usually means either:

(a) the operator was not full rank, and the solver does not support this; or

(b) the operator had a high condition number (`jnp.linalg.cond(matrix)` is large), and
    the solver can only handle low condition numbers; or

(c) the operator was not positive definite, and the solver requires positive define
    matrices.

In each case consider changing your solver to one that supports the structure of the
operator. (`optimistix.SVD()` is computationally expensive but will work for all
problems.)

Alternatively, you may have a bug in the definition of your operator. (If you were
expecting this solver to work for it.)
""".strip()


class RESULTS(metaclass=eqxi.ContainerMeta):
    successful = ""
    max_steps_reached = (
        "The maximum number of solver steps was reached. Try increasing `max_steps`."
    )
    linear_singular = _linear_singular_msg
    nonlinear_divergence = "Nonlinear solve diverged."


class Solution(eqx.Module):
    """The solution to a linear or nonlinear solve.

    **Attributes:**

    - `value`: The solution to the solve.
    - `result`: An integer representing whether the solve was successful or not. This
        can be converted into a human-readbale error message via
        `optimistix.RESULTS[result]`
    - `aux`: Any user-specified auxiliary data returned from the problem; defaults to
        `None` if there is no auxiliary data. Auxiliary outputs can be captured by
        setting a `has_aux=True` flag, e.g.
        `optx.root_findRoot(FindProblem(..., has_aux=True), ...)`.
    - `stats`: Statistics about the solver, e.g. the number of steps that were required.
    - `state`: The internal state of the solver. The meaning of this is specific to each
        solver.
    """

    value: PyTree[Array]
    result: RESULTS
    aux: PyTree[Array]
    stats: Dict[str, PyTree[Union[Array, int]]]
    state: PyTree[Any]
