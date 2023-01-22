from typing import Any, Dict

import equinox as eqx
import equinox.internal as eqxi
from jaxtyping import Array, PyTree


_linear_singular_msg = """
The linear solver returned NaN output. This usually means either:

(a) the operator was singular, and the solver requires nonsingular matrices; or

(b) the operator was not positive definite, and the solver (probably
    `optimistix.Cholesky()`) requires positive define matrices.

In each case consider changing your solver to one that supports the structure of the
operator. (`optimistix.QR()` is moderately expensive but will work for all problems.)

Alternatively, if you were expecting your operator to exhibit this structure, then you
may have a bug.
""".strip()


class RESULTS(metaclass=eqxi.ContainerMeta):
    successful = ""
    max_steps_reached = (
        "The maximum number of solver steps was reached. Try increasing `max_steps`."
    )
    linear_singular = _linear_singular_msg
    nonlinear_divergence = "Nonlinear solve diverged."


class Solution(eqx.Module):
    value: PyTree[Array]
    result: RESULTS
    state: PyTree[Any]
    aux: PyTree[Array]
    stats: Dict[str, PyTree[Array]]
