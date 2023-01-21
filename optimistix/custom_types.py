import equinox as eqx
import equinox.internal as eqxi
from jaxtyping import Array, Shaped


# Making this a Module means it typechecks against PyTree[...]
sentinel = eqxi.doc_repr(eqx.Module(), "sentinel")
Scalar = Shaped[Array, ""]
