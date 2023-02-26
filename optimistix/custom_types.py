import equinox as eqx
import equinox.internal as eqxi
import jax.tree_util as jtu
from jaxtyping import ArrayLike, Shaped


# Making this a Module means it typechecks against PyTree[...]
sentinel = eqxi.doc_repr(eqx.Module(), "sentinel")
Scalar = Shaped[ArrayLike, ""]
TreeDef = type(jtu.tree_structure(0))
