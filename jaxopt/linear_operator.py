def _is_tuple_int(x):
  return isinstance(x, tuple) and all(isinstance(y, int) for y in x)


class AbstractLinearOperator(eqx.Module):
  @abc.abstractmethod
  def mv(self, vector: PyTree[Float[Array, "_b"]]) -> PyTree[Float[Array, "_a"]]:
    ...

  @abc.abstractmethod
  def as_matrix(self) -> Float[Array, "a b"]:
    ...

  @abc.abstractmethod
  def materialise(self) -> AbstractLinearOperator:
    ...

  @abc.abstractmethod
  def in_structure(self) -> PyTree[Tuple[int, ...]]:
    ...

  @abc.abstractmethod
  def out_structure(self) -> PyTree[Tuple[int, ...]]:
    ...

  def in_size(self) -> int:
    leaves = jax.tree_leaves(self.in_structure(), is_leaf=_is_tuple_int)
    return sum(map(np.prod, leaves))

  def out_size(self) -> int:
    leaves = jax.tree_leaves(self.out_structure(), is_leaf=_is_tuple_int)
    return sum(map(np.prod, leaves))


def _matmul(matrix: Array, vector: Array) -> Array:
  # matrix has shape `(*out, *in)`
  # vector has shape `in`
  return jnp.tensordot(matrix, vector, axes=len(vector.shape))


def _tree_matmul(matrix: PyTree[Array], vector: PyTree[Array]) -> PyTree[Array]:
  # matrix has structure [tree(in), leaf(out), leaf(in)]
  # vector has structure [tree(in), leaf(in)]
  matrix = jax.tree_leaves(matrix)
  vector = jax.tree_leaves(vector)
  assert len(matrix) == len(vector)
  return sum([_matmul(m, v) for m, v in zip(matrix, vector)])


class MatrixLinearOperator(AbstractLinearOperator):
  matrix: Float[Array, "a b"]

  def mv(self, vector):
    return self.matrix @ vector

  def as_matrix(self):
    return self.matrix

  def materialise(self):
    return self

  def in_structure(self):
    in_size, _ = jnp.shape(self.matrix)
    return (in_size,)

  def out_structure(self):
    _, out_size = jnp.shape(self.matrix)
    return (out_size,)


# This is basically a generalisation of `MatrixLinearOperator` from taking
# just a single array to taking a PyTree-of-arrays.
class PyTreeLinearOperator(AbstractLinearOperator):
  pytree: PyTree[Array]
  input_structure: PyTree[Tuple[int, ...]]
  output_structure: PyTree[Tuple[int, ...]] = field(init=False)

  def __post_init__(self):
    # We are given `self.pytree` with structure [tree(out), tree(in), leaf(out), leaf(in)].
    # And we are given `input_structure` telling us [tree(in), leaf(in)].
    # This somewhat nightmarish-looking code figures out [tree(out), leaf(out)].

    in_structure = jtu.tree_structure(self.in_structure(), is_leaf=_is_tuple_int)
    is_leaf = lambda x: jtu.tree_structure(x) == in_structure

    def out_shape_impl(x, i):
      shape = jnp.shape(x)
      if len(i) == 0:
        return shape
      else:
        if shape[-len(i):] != i:
          raise ValueError("`pytree` and `input_structure` are not consistent")
        return shape[:-len(i)]

    def out_shape(x):
      if jtu.tree_structure(x) != in_structure:
        raise ValueError("`pytree` and `input_structure` are not consistent")
      shapes = jtu.tree_map(out_shape_impl, x, self.in_structure())
      if len(shapes) == 0:
        raise NotImplementedError("Cannot use `PyTreeLinearOperator` with empty `input_structure`")
      if any(shapes[0] != shape for shape in shapes[1:]):
        raise ValueError("`pytree` does not have a consistent output shape")
      return shapes[0]

    output_structure = jtu.tree_map(out_shape, self.pytree, is_leaf=is_leaf)
    object.__setattr__(self, "output_structure", output_structure)

  def mv(self, vector):
    # vector has structure [tree(in), leaf(in)]
    out_structure = jtu.tree_structure(self.output_structure, is_leaf=_is_tuple_int)
    # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
    mat_leaves = out_structure.flatten_up_to(self.pytree)
    # each of mat_leaves has structure [tree(in), leaf(out), leaf(in)]
    out_leaves = [_tree_matmul(matrix, vector) for matrix in matrix_leaves]
    # each of out_leaves has structure [leaf(out)]
    return jtu.tree_unflatten(out_structure, out_leaves)

  def as_matrix(self):
    def concat_in(shape, subpytree):
      leaves = jtu.tree_leaves(subpytree)
      assert all(jnp.shape(l)[:len(shape)] == shape for l in leaves)
      size = math.prod(shape)
      leaves = [jnp.reshape(l, (size, -1)) for l in leaves]
      return jnp.concatenate(leaves, axis=1)

    matrix = jtu.tree_map(concat_in, self.output_structure, self.pytree, is_leaf=_is_tuple_int)
    return jnp.concatenate(matrix, axis=0)

  def materialise(self):
    return self

  def in_structure(self):
    return self.input_structure

  def out_structure(self):
    return self.output_structure


class JacobianLinearOperator(AbstractLinearOperator):
  fn: Callable[[PyTree[Array["_a"]]], PyTree[Array["_b"]]]
  x: PyTree[Array["_a"]]
  args: Optional[PyTree[Array]] = None

  def mv(self, vector):
    fn = lambda x: self.fn(x, self.args)
    _, out = jax.jvp(fn, (self.x,), (vector,))
    return out

  def as_matrix(self):
    return self.materialise().as_matrix()

  def materialise(self):
    fn = lambda x: self.fn(x, self.args)
    jac = jax.jacfwd(fn)(self.x)
    return PyTreeLinearOperator(jac)

  def in_structure(self):
    return jtu.tree_map(jnp.shape, self.x)

  def out_structure(self):
    out = jax.eval_shape(self.fn, self.x, self.args)
    return jtu.tree_map(lambda o: o.shape)


class IdentityLinearOperator(eqx.Module):
  structure: PyTree[Tuple[int, ...]]

  def mv(self, vector):
    if jtu.tree_map(jnp.shape, vector) != self.in_structure():
      raise ValueError("Vector and operator structures do not match")
    return vector

  def as_matrix(self):
    return jnp.eye(self.in_size())

  def materialise(self):
    return self

  def in_structure(self):
    return self.structure

  def out_structure(self):
    return self.structure
