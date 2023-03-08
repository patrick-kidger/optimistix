from typing import Any

import equinox.internal as eqxi
from jaxtyping import ArrayLike, Shaped


sentinel: Any = eqxi.doc_repr(object(), "sentinel")
Scalar = Shaped[ArrayLike, ""]
