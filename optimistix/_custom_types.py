from typing import Any

import equinox.internal as eqxi


sentinel: Any = eqxi.doc_repr(object(), "sentinel")
