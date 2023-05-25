from typing import Any

import equinox.internal as eqxi
import jax
from typing_extensions import TypeAlias


ShapeDtypeStruct: TypeAlias = jax.ShapeDtypeStruct
sentinel: Any = eqxi.doc_repr(object(), "sentinel")
