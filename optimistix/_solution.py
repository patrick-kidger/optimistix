from typing import Any, Generic

import equinox as eqx
import lineax as lx
from jaxtyping import ArrayLike, PyTree

from ._custom_types import Aux, Y


_requires_restoration = """
The search could not find a step size above the cutoff value, or the descent failed to 
factorise the linear system. Either is signaling that a feasibility restoration is 
required to continue. (This means solving another optimisation problem to find a close
by point in the feasible set that represents an improvement over the last point.)

This is intended to be an optimistix internal result, caught and handled by the solver.
(And relevant only for constrained solves.) If you ever see this reported as the result 
of a solve, please report it as a bug here: https://github.com/patrick-kidger/optimistix 
"""


# Extend `lineax.RESULTS` as we want to be able to use their error messages too.
class RESULTS(lx.RESULTS):  # pyright: ignore
    successful = ""
    nonlinear_max_steps_reached = (
        "The maximum number of steps was reached in the nonlinear solver. "
        "The problem may not be solveable (e.g., a root-find on a function that has no "
        "roots), or you may need to increase `max_steps`."
    )
    nonlinear_divergence = "Nonlinear solve diverged."
    feasibility_restoration_required = _requires_restoration


class Solution(eqx.Module, Generic[Y, Aux]):
    """The solution to a nonlinear solve.

    **Attributes:**

    - `value`: The solution to the solve.
    - `result`: An integer representing whether the solve was successful or not. This
        can be converted into a human-readable error message via
        `optimistix.RESULTS[result]`
    - `aux`: Any user-specified auxiliary data returned from the problem; defaults to
        `None` if there is no auxiliary data. Auxiliary outputs can be captured by
        setting a `has_aux=True` flag, e.g. `optx.root_find(fn, ..., has_aux=True)`.
    - `stats`: Statistics about the solve, e.g. the number of steps that were required.
    - `state`: The final internal state of the solver. The meaning of this is specific
        to each solver.
    """

    value: Y
    result: RESULTS
    aux: Aux
    stats: dict[str, PyTree[ArrayLike]]
    state: Any
