from typing import Any

import equinox.internal as eqxi
from jaxtyping import Array, ArrayLike, Shaped


sentinel: Any = eqxi.doc_repr(object(), "sentinel")
Scalar = Shaped[Array, ""]
ScalarLike = Shaped[ArrayLike, ""]
