class Pattern(eqx.Module):
  symmetric: bool
  unit_diagonal: bool
  lower_triangular: bool
  upper_triangular: bool
  triangular: bool
  diagonal: bool

  def __init__(self, *, symmetric: bool = False, unit_diagonal: bool = False, lower_triangular: Optional[bool] = None, upper_triangular: Optional[bool] = None, triangular: Optional[bool] = None, diagonal: Optional[bool] = None):
    if lower_triangular is None:
      lower_triangular = diagonal is True
    if upper_triangular is None:
      upper_triangular = diagonal is True
    if triangular is None:
      triangular = lower_triangular or upper_triangular
    if diagonal is None:
      diagonal = lower_triangular and upper_triangular
    if symmetric and lower_triangular != upper_triangular:
      raise ValueError("A symmetric operator cannot be only lower or upper triangular")
    if triangular and not (lower_triangular or upper_triangular):
      raise ValueError("A triangular operator must be either lower trianglar or upper triangular")
    if lower_triangular or upper_triangular and not triangular:
      raise ValueError("A lower/upper triangular operator must be triangular")
    if diagonal and not (lower_triangular and upper_triangular):
      raise ValueError("A diagonal operator must be both lower and upper triangular")
    self.symmetric = symmetric
    self.unit_diagonal = unit_diagonal
    self.lower_triangular = lower_triangular
    self.upper_triangular = upper_triangular
    self.triangular = triangular
    self.diagonal = diagonal

  def transpose(self):
    if self.symmetric:
      return self
    else:
      return Pattern(symmetric=self.symmetric, unit_diagonal=self.unit_diagonal, lower_triangular=self.upper_triangular, upper_triangular=lower_triangular, triangular=self.triangular, diagonal=self.diagonal)


class AbstractLinearOperator(eqx.Module):
  def __post_init__(self):
    if self.in_size() != self.out_size():
      if self.pattern.symmetric:
        raise ValueError("Cannot have symmetric non-square operator")
      if self.pattern.diagonal:
        raise ValueError("Cannot have diagonal non-square operator")
      if self.pattern.unit_diagonal:
        raise ValueError("Cannot have non-square operator with unit diagonal")
      if self.pattern.triangular:
        raise ValueError("Cannot have triangular non-square operator")

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
  def transpose(self) -> AbstractLinearOperator:
    ...

  @abc.abstractmethod
  def in_structure(self) -> PyTree[jax.core.ShapeDtypeStruct]:
    ...

  @abc.abstractmethod
  def out_structure(self) -> PyTree[jax.core.ShapeDtypeStruct]:
    ...

  @property
  @abc.abstractmethod
  def pattern(self) -> Pattern:
    ...

  def in_size(self) -> int:
    leaves = jax.tree_leaves(self.in_structure())
    leaves = sum(np.prod(l.shape) for l in leaves)

  def out_size(self) -> int:
    leaves = jax.tree_leaves(self.out_structure())
    return sum(np.prod(l.shape) for l in leaves)


class MatrixLinearOperator(AbstractLinearOperator):
  matrix: Float[Array, "a b"]
  pattern: Pattern = Pattern()

  def mv(self, vector):
    return self.matrix @ vector

  def as_matrix(self):
    return self.matrix

  def materialise(self):
    return self

  def transpose(self):
    if self.pattern.symmetric:
      return self
    return MatrixLinearOperator(self.matrix.T, self.pattern.transpose())

  def in_structure(self):
    in_size, _ = jnp.shape(self.matrix)
    return jax.core.ShapeDtypeStruct(shape=(in_size,), dtype=self.matrix.dtype)

  def out_structure(self):
    _, out_size = jnp.shape(self.matrix)
    return jax.core.ShapeDtypeStruct(shape=(out_size,), dtype=self.matrix.dtype)


def _matmul(matrix: Array, vector: Array) -> Array:
  # matrix has structure [leaf(out), leaf(in)]
  # vector has structure [leaf(in)]
  # return has structure [leaf(out)]
  return jnp.tensordot(matrix, vector, axes=len(vector.shape))


def _tree_matmul(matrix: PyTree[Array], vector: PyTree[Array]) -> PyTree[Array]:
  # matrix has structure [tree(in), leaf(out), leaf(in)]
  # vector has structure [tree(in), leaf(in)]
  # return has structure [leaf(out)]
  matrix = jax.tree_leaves(matrix)
  vector = jax.tree_leaves(vector)
  assert len(matrix) == len(vector)
  return sum([_matmul(m, v) for m, v in zip(matrix, vector)])


# This is basically a generalisation of `MatrixLinearOperator` from taking
# just a single array to taking a PyTree-of-arrays.
class PyTreeLinearOperator(AbstractLinearOperator):
  pytree: PyTree[Array]
  output_structure: PyTree[jax.core.ShapeDtypeStruct]
  pattern: Pattern = Pattern()
  input_structure: PyTree[jax.core.ShapeDtypeStruct] = field(init=False)

  def __post_init__(self):
    # self.out_structure() has structure [tree(out)]
    # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
    def get_structure(struct, subpytree):
      # subpytree has structure [tree(in), leaf(out), leaf(in)]
      def sub_get_structure(leaf):
        shape = jnp.shape(leaf)  # [leaf(out), leaf(in)]
        ndim = len(struct.shape)
        if shape[:ndim] != struct.shape:
          raise ValueError("`pytree` and `output_structure` are not consistent")
        return jax.core.ShapeDtypeStruct(shape=shape[ndim:], dtype=jnp.dtype(leaf))
      return jtu.tree_map(sub_get_structure, subpytree)
    input_structure = jtu.tree_map(get_structure, output_structure, self.pytree)
    object.__setattr__(self, "input_structure", input_structure)

  def mv(self, vector):
    # vector has structure [tree(in), leaf(in)]
    # self.out_structure() has structure [tree(out)]
    # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
    # return has struture [tree(out), leaf(out)]
    def matmul(_, matrix):
      return _tree_matmul(matrix, vector)
    return jtu.tree_map(matmul, self.out_structure(), self.pytree)

  def as_matrix(self):
    def concat_in(struct, subpytree):
      leaves = jtu.tree_leaves(subpytree)
      assert all(jnp.shape(l)[:len(shape)] == struct.shape for l in leaves)
      size = np.prod(struct.shape)
      leaves = [jnp.reshape(l, (size, -1)) for l in leaves]
      return jnp.concatenate(leaves, axis=1)

    matrix = jtu.tree_map(concat_in, self.out_structure(), self.pytree)
    return jnp.concatenate(matrix, axis=0)

  def materialise(self):
    return self

  def transpose(self):
    if self.pattern.symmetric:
      return self
    pytree_transpose = jtu.tree_transpose(self.pytree, jax.tree_structure(self.out_structure()), jax.tree_structure(self.in_structure()))
    pytree_transpose = jtu.tree_map(jnp.transpose, pytree_transpose)
    return PyTreeLinearOperator(pytree_transpose, self.in_structure(), self.pattern.transpose())

  def in_structure(self):
    return self.input_structure

  def out_structure(self):
    return self.output_structure


class JacobianLinearOperator(AbstractLinearOperator):
  fn: Callable[[PyTree[Array["_a"]]], PyTree[Array["_b"]]]
  x: PyTree[Array["_a"]]
  args: Optional[PyTree[Array]] = None
  pattern: Pattern = Pattern()

  def mv(self, vector):
    fn = lambda x: self.fn(x, self.args)
    _, out = jax.jvp(fn, (self.x,), (vector,))
    return out

  def as_matrix(self):
    return self.materialise().as_matrix()

  def materialise(self):
    fn = lambda x: self.fn(x, self.args)
    jac = jax.jacfwd(fn)(self.x)
    return PyTreeLinearOperator(jac, self.out_structure())

  def transpose(self):
    if self.pattern.symmetric:
      return self
    return TransposeJacobianLinearOperator(self)

  def in_structure(self):
    return jax.eval_shape(lambda: self.x)

  def out_structure(self):
    return jax.eval_shape(self.fn, self.x, self.args)


class TransposeJacobianLinearOperator(AbstractLinearOperator):
  operator: JacobianLinearOperator

  def mv(self, vector):
    fn = lambda x: self.operator.fn(x, self.args)
    _, vjpfn = jax.vjp(fn, self.operator.x)
    return vjpfn(vector)

  def as_matrix(self):
    return self.operator.as_matrix().T

  def materialise(self):
    return self.operator.materialise().transpose()

  def transpose(self):
    return self.operator

  @property
  def pattern(self):
    return self.operator.pattern.transpose()

  def in_structure(self):
    return self.operator.out_structure()

  def out_structure(self):
    return self.operator.in_structure()

class IdentityLinearOperator(eqx.Module):
  structure: PyTree[jax.core.ShapeDtypeStruct]

  def mv(self, vector):
    if jax.eval_shape(lambda: vector) != self.in_structure():
      raise ValueError("Vector and operator structures do not match")
    return vector

  def as_matrix(self):
    return jnp.eye(self.in_size())

  def materialise(self):
    return self

  def transpose(self):
    return self

  def in_structure(self):
    return self.structure

  def out_structure(self):
    return self.structure

  @property
  def pattern(self):
    return Pattern(symmetric=True, unit_diagonal=True, diagonal=True)


class TangentLinearOperator(AbstractLinearOperator):
  primal: AbstractLinearOperator
  tangent: AbstractLinearOperator

  def mv(self, vector):
    mv = lambda operator: operator.mv(vector)
    _, out = eqx.filter_jvp(mv, self.primal, self.tangent)
    return out

  def as_matrix(self):
    as_matrix = lambda operator: operator.as_matrix()
    _, out = eqx.filter_jvp(as_matrix, self.primal, self.tangent)
    return out

  def materialise(self):
    materialise = lambda operator: operator.materialise()
    primal_out, tangent_out = eqx.filter_jvp(materialise), self.primal, self.tangent)
    return TangentLinearOperator(primal_out, tangent_out)

  def transpose(self):
    transpose = lambda operator: operator.transpose()
    primal_out, tangent_out = eqx.filter_jvp(transpose, self.primal, self.tangent)
    return TangentLinearOperator(primal_out, tangent_out)

  def in_structure(self):
      return self.primal.in_structure()

  def out_structure(self):
      return self.primal.out_structure()

  @property
  def pattern(self):
      return self.primal.pattern
