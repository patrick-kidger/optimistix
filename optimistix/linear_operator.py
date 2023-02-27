import abc
import functools as ft
import math
from typing import Any, Callable, FrozenSet, Iterable, List, Tuple, TypeVar, Union

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Float, PyTree, Shaped

from .custom_types import Scalar, TreeDef
from .linear_tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    nonsingular_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    transpose_tags,
    unit_diagonal_tag,
    upper_triangular_tag,
)
from .misc import cached_eval_shape, inexact_asarray, jacobian, NoneAux


def _frozenset(x: Union[object, Iterable[object]]) -> FrozenSet[object]:
    try:
        iter_x = iter(x)
    except TypeError:
        return frozenset([x])
    else:
        return frozenset(iter_x)


class AbstractLinearOperator(eqx.Module):
    @abc.abstractmethod
    def mv(self, vector: PyTree[Float[Array, " _b"]]) -> PyTree[Float[Array, " _a"]]:
        ...

    @abc.abstractmethod
    def as_matrix(self) -> Float[Array, "a b"]:
        ...

    @abc.abstractmethod
    def transpose(self) -> "AbstractLinearOperator":
        ...

    @abc.abstractmethod
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        ...

    @abc.abstractmethod
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        ...

    def in_size(self) -> int:
        leaves = jtu.tree_leaves(self.in_structure())
        return sum(math.prod(leaf.shape) for leaf in leaves)

    def out_size(self) -> int:
        leaves = jtu.tree_leaves(self.out_structure())
        return sum(math.prod(leaf.shape) for leaf in leaves)

    @property
    def T(self):
        return self.transpose()

    def __add__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only add AbstractLinearOperators together.")
        return AddLinearOperator(self, other)

    def __mul__(self, other):
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError("Can only multiply AbstractLinearOperators by scalars.")
        return MulLinearOperator(self, other)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only compose AbstractLinearOperators together.")
        return ComposedLinearOperator(self, other)

    def __truediv__(self, other):
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError("Can only divide AbstractLinearOperators by scalars.")
        return DivLinearOperator(self, other)


class MatrixLinearOperator(AbstractLinearOperator):
    matrix: Float[Array, "a b"]
    tags: FrozenSet[object]

    def __init__(
        self, matrix: Shaped[Array, "a b"], tags: Union[object, FrozenSet[object]] = ()
    ):
        if jnp.ndim(matrix) != 2:
            raise ValueError(
                "`MatrixLinearOperator(matrix=...)` should be 2-dimensional."
            )
        if not jnp.issubdtype(matrix, jnp.inexact):
            matrix = matrix.astype(jnp.float32)
        self.matrix = matrix
        self.tags = _frozenset(tags)

    def mv(self, vector):
        return jnp.matmul(self.matrix, vector, precision=lax.Precision.HIGHEST)

    def as_matrix(self):
        return self.matrix

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        return MatrixLinearOperator(self.matrix.T, transpose_tags(self.tags))

    def in_structure(self):
        in_size, _ = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(in_size,), dtype=self.matrix.dtype)

    def out_structure(self):
        _, out_size = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(out_size,), dtype=self.matrix.dtype)


def _matmul(matrix: ArrayLike, vector: ArrayLike) -> Array:
    # matrix has structure [leaf(out), leaf(in)]
    # vector has structure [leaf(in)]
    # return has structure [leaf(out)]
    return jnp.tensordot(
        matrix, vector, axes=jnp.ndim(vector), precision=lax.Precision.HIGHEST
    )


def _tree_matmul(matrix: PyTree[ArrayLike], vector: PyTree[ArrayLike]) -> PyTree[Array]:
    # matrix has structure [tree(in), leaf(out), leaf(in)]
    # vector has structure [tree(in), leaf(in)]
    # return has structure [leaf(out)]
    matrix = jtu.tree_leaves(matrix)
    vector = jtu.tree_leaves(vector)
    assert len(matrix) == len(vector)
    return sum([_matmul(m, v) for m, v in zip(matrix, vector)])


# Needed as static fields must be hashable and eq-able, and custom pytrees might have
# e.g. define custom __eq__ methods.
_T = TypeVar("_T")
_FlatPyTree = Tuple[List[_T], TreeDef]


def _inexact_structure_impl2(x):
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return x
    else:
        return x.astype(jnp.float32)


def _inexact_structure_impl(x):
    return jtu.tree_map(_inexact_structure_impl2, x)


def _inexact_structure(x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
    return jax.eval_shape(_inexact_structure_impl, x)


class _Leaf:  # not a pytree
    def __init__(self, value):
        self.value = value


# This is basically a generalisation of `MatrixLinearOperator` from taking
# just a single array to taking a PyTree-of-arrays.
# The `{input,output}_structure`s have to be static because otherwise abstract
# evaluation rules will promote them to ShapedArrays.
class PyTreeLinearOperator(AbstractLinearOperator):
    pytree: PyTree[Float[Array, "..."]]
    output_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.static_field()
    tags: FrozenSet[object]
    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.static_field()

    def __init__(
        self,
        pytree: PyTree[ArrayLike],
        output_structure: PyTree[jax.ShapeDtypeStruct],
        tags: Union[object, FrozenSet[object]] = (),
    ):
        output_structure = _inexact_structure(output_structure)
        self.pytree = jtu.tree_map(inexact_asarray, pytree)
        self.output_structure = jtu.tree_flatten(output_structure)
        self.tags = _frozenset(tags)

        # self.out_structure() has structure [tree(out)]
        # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
        def get_structure(struct, subpytree):
            # subpytree has structure [tree(in), leaf(out), leaf(in)]
            def sub_get_structure(leaf):
                shape = jnp.shape(leaf)  # [leaf(out), leaf(in)]
                ndim = len(struct.shape)
                if shape[:ndim] != struct.shape:
                    raise ValueError(
                        "`pytree` and `output_structure` are not consistent"
                    )
                return jax.ShapeDtypeStruct(shape=shape[ndim:], dtype=jnp.dtype(leaf))

            return _Leaf(jtu.tree_map(sub_get_structure, subpytree))

        if output_structure is None:
            # Implies that len(input_structures) > 0
            raise ValueError("Cannot have trivial output_structure")
        input_structures = jtu.tree_map(get_structure, output_structure, self.pytree)
        input_structures = jtu.tree_leaves(input_structures)
        input_structure = input_structures[0].value
        for val in input_structures[1:]:
            if eqx.tree_equal(input_structure, val.value) is not True:
                raise ValueError(
                    "`pytree` does not have a consistent `input_structure`"
                )
        self.input_structure = jtu.tree_flatten(input_structure)

    def mv(self, vector):
        # vector has structure [tree(in), leaf(in)]
        # self.out_structure() has structure [tree(out)]
        # self.pytree has structure [tree(out), tree(in), leaf(out), leaf(in)]
        # return has struture [tree(out), leaf(out)]
        def matmul(_, matrix):
            return _tree_matmul(matrix, vector)

        return jtu.tree_map(matmul, self.out_structure(), self.pytree)

    def as_matrix(self):
        dtype = jnp.result_type(*jtu.tree_leaves(self.pytree))

        def concat_in(struct, subpytree):
            leaves = jtu.tree_leaves(subpytree)
            assert all(
                leaf.shape[: len(struct.shape)] == struct.shape for leaf in leaves
            )
            size = math.prod(struct.shape)
            leaves = [leaf.astype(dtype).reshape(size, -1) for leaf in leaves]
            return jnp.concatenate(leaves, axis=1)

        matrix = jtu.tree_map(concat_in, self.out_structure(), self.pytree)
        matrix = jtu.tree_leaves(matrix)
        return jnp.concatenate(matrix, axis=0)

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        pytree_transpose = jtu.tree_transpose(
            jtu.tree_structure(self.out_structure()),
            jtu.tree_structure(self.in_structure()),
            self.pytree,
        )
        pytree_transpose = jtu.tree_map(jnp.transpose, pytree_transpose)
        return PyTreeLinearOperator(
            pytree_transpose, self.in_structure(), transpose_tags(self.tags)
        )

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        leaves, treedef = self.output_structure
        return jtu.tree_unflatten(treedef, leaves)


class _NoAuxIn(eqx.Module):
    fn: Callable
    args: PyTree[Array]

    def __call__(self, x):
        return self.fn(x, self.args)


class _NoAuxOut(eqx.Module):
    fn: Callable

    def __call__(self, x):
        f, _ = self.fn(x)
        return f


class _Unwrap(eqx.Module):
    fn: Callable

    def __call__(self, x):
        (f,) = self.fn(x)
        return f


class JacobianLinearOperator(AbstractLinearOperator):
    fn: Callable[
        [PyTree[Float[Array, "..."]], PyTree[Any]], PyTree[Float[Array, "..."]]
    ]
    x: PyTree[Float[Array, "..."]]
    args: PyTree[Any]
    tags: FrozenSet[object]

    def __init__(
        self,
        fn: Callable,
        x: PyTree[ArrayLike],
        args: PyTree[Any] = None,
        tags: Union[object, Iterable[object]] = (),
        _has_aux: bool = False,
    ):
        if not _has_aux:
            fn = NoneAux(fn)
        # Flush out any closed-over values, so that we can safely pass `self`
        # across API boundaries. (In particular, across `linear_solve_p`.)
        # We don't use `jax.closure_convert` as that only flushes autodiffable
        # (=floating-point) constants. It probably doesn't matter, but if `fn` is a
        # PyTree capturing non-floating-point constants, we should probably continue
        # to respect that, and keep any non-floating-point constants as part of the
        # PyTree structure.
        x = jtu.tree_map(inexact_asarray, x)
        fn = eqx.filter_closure_convert(fn, x, args)
        self.fn = fn
        self.x = x
        self.args = args
        self.tags = _frozenset(tags)

    def mv(self, vector):
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        _, out = jax.jvp(fn, (self.x,), (vector,))
        return out

    def as_matrix(self):
        return materialise(self).as_matrix()

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        # Works because vjpfn is a PyTree
        _, vjpfn = jax.vjp(fn, self.x)
        vjpfn = _Unwrap(vjpfn)
        return FunctionLinearOperator(
            vjpfn, self.out_structure(), transpose_tags(self.tags)
        )

    def in_structure(self):
        return jax.eval_shape(lambda: self.x)

    def out_structure(self):
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        return cached_eval_shape(fn, self.x)


# `input_structure` must be static as with `JacobianLinearOperator`
class FunctionLinearOperator(AbstractLinearOperator):
    fn: Callable[[PyTree[Float[Array, "..."]]], PyTree[Float[Array, "..."]]]
    input_structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.static_field()
    tags: FrozenSet[object]

    def __init__(
        self,
        fn: Callable[[PyTree[Float[Array, "..."]]], PyTree[Float[Array, "..."]]],
        input_structure: PyTree[jax.ShapeDtypeStruct],
        tags: Union[object, Iterable[object]] = (),
    ):
        # See matching comment in JacobianLinearOperator.
        fn = eqx.filter_closure_convert(fn, input_structure)
        input_structure = _inexact_structure(input_structure)
        self.fn = fn
        self.input_structure = jtu.tree_flatten(input_structure)
        self.tags = _frozenset(tags)

    def mv(self, vector):
        return self.fn(vector)

    def as_matrix(self):
        return materialise(self).as_matrix()

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        transpose_fn = jax.linear_transpose(self.fn, self.in_structure())

        def _transpose_fn(vector):
            (out,) = transpose_fn(vector)
            return out

        # Works because transpose_fn is a PyTree
        return FunctionLinearOperator(
            _transpose_fn, self.out_structure(), transpose_tags(self.tags)
        )

    def in_structure(self):
        leaves, treedef = self.input_structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        return cached_eval_shape(self.fn, self.in_structure())


# `structure` must be static as with `JacobianLinearOperator`
class IdentityLinearOperator(AbstractLinearOperator):
    structure: _FlatPyTree[jax.ShapeDtypeStruct] = eqx.static_field()

    def __init__(self, structure: PyTree[jax.ShapeDtypeStruct]):
        structure = _inexact_structure(structure)
        self.structure = jtu.tree_flatten(structure)

    def mv(self, vector):
        if jax.eval_shape(lambda: vector) != self.in_structure():
            raise ValueError("Vector and operator structures do not match")
        return vector

    def as_matrix(self):
        return jnp.eye(self.in_size())

    def transpose(self):
        return self

    def in_structure(self):
        leaves, treedef = self.structure
        return jtu.tree_unflatten(treedef, leaves)

    def out_structure(self):
        leaves, treedef = self.structure
        return jtu.tree_unflatten(treedef, leaves)

    @property
    def tags(self):
        return frozenset()


class DiagonalLinearOperator(AbstractLinearOperator):
    diagonal: Float[Array, " size"]

    def __init__(self, diagonal: Shaped[Array, " size"]):
        self.diagonal = inexact_asarray(diagonal)

    def mv(self, vector):
        return self.diagonal * vector

    def as_matrix(self):
        return jnp.diag(self.diagonal)

    def transpose(self):
        return self

    def in_structure(self):
        (size,) = jnp.shape(self.diagonal)
        return jax.ShapeDtypeStruct(shape=(size,), dtype=self.diagonal.dtype)

    def out_structure(self):
        (size,) = jnp.shape(self.diagonal)
        return jax.ShapeDtypeStruct(shape=(size,), dtype=self.diagonal.dtype)


class TaggedLinearOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    tags: FrozenSet[object]

    def __init__(
        self, operator: AbstractLinearOperator, tags: Union[object, Iterable[object]]
    ):
        self.operator = operator
        self.tags = _frozenset(tags)

    def mv(self, vector):
        return self.operator.mv(vector)

    def as_matrix(self):
        return self.operator.as_matrix()

    def transpose(self):
        return TaggedLinearOperator(
            self.operator.transpose(), transpose_tags(self.tags)
        )

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


#
# All operators below here are private to Optimistix.
#


class TangentLinearOperator(AbstractLinearOperator):
    primal: AbstractLinearOperator
    tangent: AbstractLinearOperator

    def __post_init__(self):
        assert type(self.primal) is type(self.tangent)  # noqa: E721
        assert self.primal.in_structure() == self.tangent.in_structure()
        assert self.primal.out_structure() == self.tangent.out_structure()

    def mv(self, vector):
        mv = lambda operator: operator.mv(vector)
        _, out = eqx.filter_jvp(mv, self.primal, self.tangent)
        return out

    def as_matrix(self):
        as_matrix = lambda operator: operator.as_matrix()
        _, out = eqx.filter_jvp(as_matrix, self.primal, self.tangent)
        return out

    def transpose(self):
        transpose = lambda operator: operator.transpose()
        primal_out, tangent_out = eqx.filter_jvp(transpose, self.primal, self.tangent)
        return TangentLinearOperator(primal_out, tangent_out)

    def in_structure(self):
        return self.primal.in_structure()

    def out_structure(self):
        return self.primal.out_structure()


class AddLinearOperator(AbstractLinearOperator):
    operator1: AbstractLinearOperator
    operator2: AbstractLinearOperator

    def __post_init__(self):
        if self.operator1.in_structure() != self.operator2.in_structure():
            raise ValueError("Incompatible linear operator structures")
        if self.operator1.out_structure() != self.operator2.out_structure():
            raise ValueError("Incompatible linear operator structures")

    def mv(self, vector):
        mv1 = self.operator1.mv(vector)
        mv2 = self.operator2.mv(vector)
        return (mv1**ω + mv2**ω).ω

    def as_matrix(self):
        return self.operator1.as_matrix() + self.operator2.as_matrix()

    def transpose(self):
        return self.operator1.transpose() + self.operator2.transpose()

    def in_structure(self):
        return self.operator1.in_structure()

    def out_structure(self):
        return self.operator1.out_structure()


class MulLinearOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    scalar: Scalar

    def mv(self, vector):
        return (self.operator.mv(vector) ** ω * self.scalar).ω

    def as_matrix(self):
        return self.operator.as_matrix() * self.scalar

    def transpose(self):
        return self.operator.transpose() * self.scalar

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


class DivLinearOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    scalar: Scalar

    def mv(self, vector):
        return (self.operator.mv(vector) ** ω / self.scalar).ω

    def as_matrix(self):
        return self.operator.as_matrix() / self.scalar

    def transpose(self):
        return self.operator.transpose() / self.scalar

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


class ComposedLinearOperator(AbstractLinearOperator):
    operator1: AbstractLinearOperator
    operator2: AbstractLinearOperator

    def __post_init__(self):
        if self.operator1.in_structure() != self.operator2.out_structure():
            raise ValueError("Incompatible linear operator structures")

    def mv(self, vector):
        return self.operator1.mv(self.operator2.mv(vector))

    def as_matrix(self):
        return jnp.matmul(
            self.operator1.as_matrix(),
            self.operator2.as_matrix(),
            precision=lax.Precision.HIGHEST,
        )

    def transpose(self):
        return self.operator2.transpose() @ self.operator1.transpose()

    def in_structure(self):
        return self.operator1.in_structure()

    def out_structure(self):
        return self.operator2.out_structure()


class AuxLinearOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    aux: PyTree[Array]

    def mv(self, vector):
        return self.operator.mv(vector)

    def as_matrix(self):
        return self.operator.as_matrix()

    def transpose(self):
        return self.operator.transpose()

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()


#
# Operations on `AbstractLinearOperator`s.
# These are done through `singledispatch` rather than as methods.
#
# If an end user ever wanted to add something analogous to
# `diagonal: AbstractLinearOperator -> Array`
# then of course they don't get to edit our base class and add overloads to all
# subclasses.
# They'd have to use `singledispatch` to get the desired behaviour. (Or maybe just
# hardcode compatibility with only some `AbstractLinearOperator` subclasses, eurgh.)
# So for consistency we do the same thing here, rather than adding privileged behaviour
# for just the operations we happen to support.
#
# (Something something Julia something something orphan problem etc.)
#


def _default_not_implemented(name: str, operator: AbstractLinearOperator):
    msg = f"`optimistix.{name}` has not been implemented for {type(operator)}"
    if type(operator).__module__.startswith("optimistix"):
        assert False, msg + ". Please file a bug against Optimistix."
    else:
        raise NotImplementedError(msg)


# linearise


@ft.singledispatch
def linearise(operator: AbstractLinearOperator) -> AbstractLinearOperator:
    _default_not_implemented("linearise", operator)


@linearise.register(MatrixLinearOperator)
@linearise.register(PyTreeLinearOperator)
@linearise.register(FunctionLinearOperator)
@linearise.register(IdentityLinearOperator)
@linearise.register(DiagonalLinearOperator)
def _(operator):
    return operator


@linearise.register(JacobianLinearOperator)
def _(operator):
    fn = _NoAuxIn(operator.fn, operator.args)
    (_, aux), lin = jax.linearize(fn, operator.x)
    lin = _NoAuxOut(lin)
    out = FunctionLinearOperator(lin, operator.in_structure(), operator.tags)
    return AuxLinearOperator(out, aux)


# materialise


@ft.singledispatch
def materialise(operator: AbstractLinearOperator) -> AbstractLinearOperator:
    _default_not_implemented("materialise", operator)


@materialise.register(MatrixLinearOperator)
@materialise.register(PyTreeLinearOperator)
@materialise.register(IdentityLinearOperator)
@materialise.register(DiagonalLinearOperator)
def _(operator):
    return operator


@materialise.register(JacobianLinearOperator)
def _(operator):
    fn = _NoAuxIn(operator.fn, operator.args)
    jac, aux = jacobian(fn, operator.in_size(), operator.out_size(), has_aux=True)(
        operator.x
    )
    out = PyTreeLinearOperator(jac, operator.out_structure(), operator.tags)
    return AuxLinearOperator(out, aux)


@materialise.register(FunctionLinearOperator)
def _(operator):
    # TODO(kidger): implement more efficiently, without the relinearisation
    zeros = jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), operator.in_structure())
    jac = jacobian(operator.fn, operator.in_size(), operator.out_size())(zeros)
    return PyTreeLinearOperator(jac, operator.out_structure(), operator.tags)


# diagonal


@ft.singledispatch
def diagonal(operator: AbstractLinearOperator) -> Shaped[Array, " size"]:
    _default_not_implemented("diagonal", operator)


@diagonal.register(MatrixLinearOperator)
@diagonal.register(PyTreeLinearOperator)
@diagonal.register(JacobianLinearOperator)
@diagonal.register(FunctionLinearOperator)
def _(operator):
    return jnp.diag(operator.as_matrix())


@diagonal.register(IdentityLinearOperator)
def _(operator):
    return jnp.ones(operator.in_size())


@diagonal.register(DiagonalLinearOperator)
def _(operator):
    return operator.diagonal


# is_symmetric


@ft.singledispatch
def is_symmetric(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("is_symmetric", operator)


@is_symmetric.register(MatrixLinearOperator)
@is_symmetric.register(PyTreeLinearOperator)
@is_symmetric.register(JacobianLinearOperator)
@is_symmetric.register(FunctionLinearOperator)
def _(operator):
    return symmetric_tag in operator.tags


@is_symmetric.register(IdentityLinearOperator)
@is_symmetric.register(DiagonalLinearOperator)
def _(operator):
    return True


# is_diagonal


@ft.singledispatch
def is_diagonal(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("is_diagonal", operator)


@is_diagonal.register(MatrixLinearOperator)
@is_diagonal.register(PyTreeLinearOperator)
@is_diagonal.register(JacobianLinearOperator)
@is_diagonal.register(FunctionLinearOperator)
def _(operator):
    return diagonal_tag in operator.tags


@is_diagonal.register(IdentityLinearOperator)
@is_diagonal.register(DiagonalLinearOperator)
def _(operator):
    return True


# has_unit_diagonal


@ft.singledispatch
def has_unit_diagonal(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("has_unit_diagonal", operator)


@has_unit_diagonal.register(MatrixLinearOperator)
@has_unit_diagonal.register(PyTreeLinearOperator)
@has_unit_diagonal.register(JacobianLinearOperator)
@has_unit_diagonal.register(FunctionLinearOperator)
def _(operator):
    return unit_diagonal_tag in operator.tags


@has_unit_diagonal.register(IdentityLinearOperator)
def _(operator):
    return True


@has_unit_diagonal.register(DiagonalLinearOperator)
def _(operator):
    # TODO: refine this
    return False


# is_lower_triangular


@ft.singledispatch
def is_lower_triangular(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("is_lower_triangular", operator)


@is_lower_triangular.register(MatrixLinearOperator)
@is_lower_triangular.register(PyTreeLinearOperator)
@is_lower_triangular.register(JacobianLinearOperator)
@is_lower_triangular.register(FunctionLinearOperator)
def _(operator):
    return lower_triangular_tag in operator.tags


@is_lower_triangular.register(IdentityLinearOperator)
@is_lower_triangular.register(DiagonalLinearOperator)
def _(operator):
    return True


# is_upper_triangular


@ft.singledispatch
def is_upper_triangular(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("is_upper_triangular", operator)


@is_upper_triangular.register(MatrixLinearOperator)
@is_upper_triangular.register(PyTreeLinearOperator)
@is_upper_triangular.register(JacobianLinearOperator)
@is_upper_triangular.register(FunctionLinearOperator)
def _(operator):
    return upper_triangular_tag in operator.tags


@is_upper_triangular.register(IdentityLinearOperator)
@is_upper_triangular.register(DiagonalLinearOperator)
def _(operator):
    return True


# is_positive_semidefinite


@ft.singledispatch
def is_positive_semidefinite(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("is_positive_semidefinite", operator)


@is_positive_semidefinite.register(MatrixLinearOperator)
@is_positive_semidefinite.register(PyTreeLinearOperator)
@is_positive_semidefinite.register(JacobianLinearOperator)
@is_positive_semidefinite.register(FunctionLinearOperator)
def _(operator):
    return positive_semidefinite_tag in operator.tags


@is_positive_semidefinite.register(IdentityLinearOperator)
def _(operator):
    return True


@is_positive_semidefinite.register(DiagonalLinearOperator)
def _(operator):
    # TODO: refine this
    return False


# is_negative_semidefinite


@ft.singledispatch
def is_negative_semidefinite(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("is_negative_semidefinite", operator)


@is_negative_semidefinite.register(MatrixLinearOperator)
@is_negative_semidefinite.register(PyTreeLinearOperator)
@is_negative_semidefinite.register(JacobianLinearOperator)
@is_negative_semidefinite.register(FunctionLinearOperator)
def _(operator):
    return negative_semidefinite_tag in operator.tags


@is_negative_semidefinite.register(IdentityLinearOperator)
def _(operator):
    return False


@is_negative_semidefinite.register(DiagonalLinearOperator)
def _(operator):
    # TODO: refine this
    return False


# is_nonsingular


@ft.singledispatch
def is_nonsingular(operator: AbstractLinearOperator) -> bool:
    _default_not_implemented("is_nonsingular", operator)


@is_nonsingular.register(MatrixLinearOperator)
@is_nonsingular.register(PyTreeLinearOperator)
@is_nonsingular.register(JacobianLinearOperator)
@is_nonsingular.register(FunctionLinearOperator)
def _(operator):
    return nonsingular_tag in operator.tags


@is_nonsingular.register(IdentityLinearOperator)
def _(operator):
    return True


@is_nonsingular.register(DiagonalLinearOperator)
def _(operator):
    # TODO: refine this
    return False


# ops for wrapper operators


@linearise.register(TaggedLinearOperator)
def _(operator):
    return TaggedLinearOperator(linearise(operator.operator), operator.tags)


@materialise.register(TaggedLinearOperator)
def _(operator):
    return TaggedLinearOperator(materialise(operator.operator), operator.tags)


@diagonal.register(TaggedLinearOperator)
def _(operator):
    # Untagged; we might not have any of the properties our tags represent any more.
    return diagonal(operator.operator)


for transform in (linearise, materialise, diagonal):

    @transform.register(TangentLinearOperator)
    def _(operator, transform=transform):
        primal_out, tangent_out = eqx.filter_jvp(
            transform, operator.primal, operator.tangent
        )
        return TangentLinearOperator(primal_out, tangent_out)

    @transform.register(AddLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator1) + transform(operator.operator2)

    @transform.register(MulLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator) * operator.scalar

    @transform.register(DivLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator) / operator.scalar

    @transform.register(AuxLinearOperator)
    def _(operator, transform=transform):
        return transform(operator.operator)


@linearise.register(ComposedLinearOperator)
def _(operator):
    return linearise(operator.operator1) @ linearise(operator.operator2)


@materialise.register(ComposedLinearOperator)
def _(operator):
    return materialise(operator.operator1) @ materialise(operator.operator2)


@diagonal.register(ComposedLinearOperator)
def _(operator):
    return jnp.diag(operator.as_matrix())


for check in (
    is_symmetric,
    is_diagonal,
    has_unit_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_positive_semidefinite,
    is_negative_semidefinite,
    is_nonsingular,
):

    @check.register(TangentLinearOperator)
    def _(operator, check=check):
        return check(operator.primal)

    @check.register(MulLinearOperator)
    @check.register(DivLinearOperator)
    @check.register(AuxLinearOperator)
    def _(operator, check=check):
        return check(operator.operator)


for check, tag in (
    (is_symmetric, symmetric_tag),
    (is_diagonal, diagonal_tag),
    (has_unit_diagonal, unit_diagonal_tag),
    (is_lower_triangular, lower_triangular_tag),
    (is_upper_triangular, upper_triangular_tag),
    (is_positive_semidefinite, positive_semidefinite_tag),
    (is_negative_semidefinite, negative_semidefinite_tag),
    (is_nonsingular, nonsingular_tag),
):

    @check.register(TaggedLinearOperator)
    def _(operator, check=check, tag=tag):
        return (tag in operator.tags) or check(operator.operator)


for check in (
    is_symmetric,
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_positive_semidefinite,
    is_negative_semidefinite,
):

    @check.register(AddLinearOperator)
    def _(operator, check=check):
        return check(operator.operator1) and check(operator.operator2)


@has_unit_diagonal.register(AddLinearOperator)
def _(operator):
    return False


@is_nonsingular.register(AddLinearOperator)
def _(operator):
    return False


for check in (
    is_symmetric,
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_positive_semidefinite,
    is_negative_semidefinite,
    is_nonsingular,
):

    @check.register(ComposedLinearOperator)
    def _(operator, check=check):
        return check(operator.operator1) and check(operator.operator2)


@has_unit_diagonal.register(ComposedLinearOperator)
def _(operator):
    a = is_diagonal(operator)
    b = is_lower_triangular(operator)
    c = is_upper_triangular(operator)
    d = has_unit_diagonal(operator.operator1)
    e = has_unit_diagonal(operator.operator2)
    return (a or b or c) and d and e
