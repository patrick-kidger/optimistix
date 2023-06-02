from typing import Any, Union

import equinox as eqx
import lineax as lx
from jaxtyping import Array, PyTree


# Extend `lineax.RESULTS` as we want to be able to use their error messages too.
class RESULTS(lx.RESULTS):  # pyright: ignore
    successful = ""
    max_steps_reached = (
        "The maximum number of solver steps was reached. Try increasing `max_steps`."
    )
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
    stats: dict[str, PyTree[Union[Array, int]]]
    state: PyTree[Any]
