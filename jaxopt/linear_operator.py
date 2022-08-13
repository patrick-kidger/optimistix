def _is_tuple_int(x):
  return isinstance(x, tuple) and all(isinstance(y, int) for y in x)


class AbstractLinearOperator(eqx.Module):
  @abc.abstractmethod
  def as_matrix(self) -> Array["a b"]:
    ...

  @abc.abstractmethod
  def mv(self, vector: Array["a"]) -> Array["b"]:
    ...

  @abc.abstractmethod
  def materialise(self) -> LinearOperator:
    ...

  @abc.abstractmethod
  def in_structure(self) -> PyTree[Tuple[int, ...]]:
    ...

  @abc.abstractmethod
  def out_structure(self) -> PyTree[Tuple[int, ...]]:
    ...

  def in_size(self):
    leaves = jax.tree_leaves(self.in_structure(), is_leaf=_is_tuple_int)
    return sum(map(np.prod, leaves))

  def out_size(self):
    leaves = jax.tree_leaves(self.out_structure(), is_leaf=_is_tuple_int)
    return sum(map(np.prod, leaves))


class MatrixLinearOperator(AbstractLinearOperator):
  matrix: Array["a b"]

  def as_matrix(self):
    return self.matrix

  def mv(self, vector):
    return self.matrix @ vector

  def materialise(self):
    return self

  def in_structure(self):
    in_size, _ = self.matrix.shape
    return (in_size,)

  def out_structure(self):
    _, out_size = self.matrix.shape
    return (out_size,)


class JacobianLinearOperator(AbstractLinearOperator):
  fn: Callable[[Array["a"]], Array["b"]]
  x: Array["a"]

  def as_matrix(self):
    return jax.jacfwd(self.fn)(self.x)

  def mv(self, vector):
    _, out = jax.jvp(self.fn, (self.x,), (vector,))
    return out

  def materialise(self):
    return MatrixLinearOperator(self.as_matrix())

  def in_structure(self):
    assert isinstance(self.x, jnp.ndarray)
    return x.shape

  def out_structure(self):
    assert isinstance(self.x, jnp.ndarray)
    out = jax.eval_shape(fn, x)
    assert isinstance(out, jax.core.ShapeDtypeStruct)
    return out.shape


class IdentityLinearOperator(eqx.Module):
  structure: PyTree[Tuple[int, ...]]

  def as_matrix(self):
    return jnp.eye(self.in_size())

  def mv(self, vector):
    return vector

  def materialise(self):
    return self

  def in_structure(self):
    return self.structure

  def out_structure(self):
    return self.structure
