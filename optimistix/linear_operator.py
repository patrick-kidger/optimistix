import abc
from dataclasses import field
from typing import Any, Callable, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, PyTree, Shaped

from .custom_types import Scalar
from .misc import cached_eval_shape, jacobian, NoneAux


# `False` indicates "unknown" (that we can't rely on exploiting this property), not that
# the property definitively doesn't hold.
class Pattern(eqx.Module):
    symmetric: bool = False
    unit_diagonal: bool = False
    lower_triangular: bool = False
    upper_triangular: bool = False
    diagonal: bool = False
    positive_semidefinite: bool = False
    negative_semidefinite: bool = False
    nonsingular: bool = False

    def transpose(self):
        return Pattern(
            symmetric=self.symmetric,
            unit_diagonal=self.unit_diagonal,
            lower_triangular=self.upper_triangular,
            upper_triangular=self.lower_triangular,
            diagonal=self.diagonal,
            positive_semidefinite=self.positive_semidefinite,
            negative_semidefinite=self.negative_semidefinite,
            nonsingular=self.nonsingular,
        )


class AbstractLinearOperator(eqx.Module):
    pattern = eqxi.abstractattribute(Pattern)

    def __post_init__(self):
        if self.in_size() != self.out_size():
            if self.pattern not in (Pattern(), Pattern(nonsingular=True)):
                raise ValueError(
                    f"Cannot have a non-square operator with pattern {self.pattern}"
                )

    @abc.abstractmethod
    def mv(self, vector: PyTree[Shaped[Array, " _b"]]) -> PyTree[Shaped[Array, " _a"]]:
        ...

    @abc.abstractmethod
    def as_matrix(self) -> Shaped[Array, "a b"]:
        ...

    @abc.abstractmethod
    def materialise(self) -> "AbstractLinearOperator":
        ...

    @abc.abstractmethod
    def linearise(self) -> "AbstractLinearOperator":
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
        return sum(np.prod(leaf.shape).item() for leaf in leaves)

    def out_size(self) -> int:
        leaves = jtu.tree_leaves(self.out_structure())
        return sum(np.prod(leaf.shape).item() for leaf in leaves)

    @property
    def T(self):
        return self.transpose()

    def __add__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only add AbstractLinearOperators together.")
        return _AddLinearOperator(self, other)

    def __mul__(self, other):
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError("Can only multiply AbstractLinearOperators by scalars.")
        return _MulLinearOperator(self, other)

    def __matmul__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            raise ValueError("Can only compose AbstractLinearOperators together.")
        return _ComposedLinearOperator(self, other)


class MatrixLinearOperator(AbstractLinearOperator):
    matrix: Shaped[Array, "a b"]
    pattern: Pattern = Pattern()

    def mv(self, vector):
        return jnp.matmul(self.matrix, vector, precision=lax.Precision.HIGHEST)

    def as_matrix(self):
        return self.matrix

    def materialise(self):
        return self

    def linearise(self):
        return self

    def transpose(self):
        if self.pattern.symmetric:
            return self
        return MatrixLinearOperator(self.matrix.T, self.pattern.transpose())

    def in_structure(self):
        in_size, _ = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(in_size,), dtype=self.matrix.dtype)

    def out_structure(self):
        _, out_size = jnp.shape(self.matrix)
        return jax.ShapeDtypeStruct(shape=(out_size,), dtype=self.matrix.dtype)


def _matmul(matrix: Array, vector: Array) -> Array:
    # matrix has structure [leaf(out), leaf(in)]
    # vector has structure [leaf(in)]
    # return has structure [leaf(out)]
    return jnp.tensordot(
        matrix, vector, axes=jnp.ndim(vector), precision=lax.Precision.HIGHEST
    )


def _tree_matmul(matrix: PyTree[Array], vector: PyTree[Array]) -> PyTree[Array]:
    # matrix has structure [tree(in), leaf(out), leaf(in)]
    # vector has structure [tree(in), leaf(in)]
    # return has structure [leaf(out)]
    matrix = jtu.tree_leaves(matrix)
    vector = jtu.tree_leaves(vector)
    assert len(matrix) == len(vector)
    return sum([_matmul(m, v) for m, v in zip(matrix, vector)])


# This is basically a generalisation of `MatrixLinearOperator` from taking
# just a single array to taking a PyTree-of-arrays.
# The `{input,output}_structure`s have to be static because otherwise abstract
# evaluation rules will promote them to ShapedArrays.
class PyTreeLinearOperator(AbstractLinearOperator):
    pytree: PyTree[Array]
    output_structure: PyTree[jax.ShapeDtypeStruct] = eqx.static_field()
    pattern: Pattern = Pattern()
    input_structure: PyTree[jax.ShapeDtypeStruct] = eqx.static_field(init=False)

    def __post_init__(self):
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

            return jtu.tree_map(sub_get_structure, subpytree)

        input_structure = jtu.tree_map(
            get_structure, self.output_structure, self.pytree
        )
        self.input_structure = input_structure

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
            assert all(
                jnp.shape(leaf)[: len(struct.shape)] == struct.shape for leaf in leaves
            )
            size = np.prod(struct.shape)
            leaves = [jnp.reshape(leaf, (size, -1)) for leaf in leaves]
            return jnp.concatenate(leaves, axis=1)

        matrix = jtu.tree_map(concat_in, self.out_structure(), self.pytree)
        matrix = jtu.tree_leaves(matrix)
        return jnp.concatenate(matrix, axis=0)

    def materialise(self):
        return self

    def linearise(self):
        return self

    def transpose(self):
        if self.pattern.symmetric:
            return self
        pytree_transpose = jtu.tree_transpose(
            jtu.tree_structure(self.out_structure()),
            jtu.tree_structure(self.in_structure()),
            self.pytree,
        )
        pytree_transpose = jtu.tree_map(jnp.transpose, pytree_transpose)
        return PyTreeLinearOperator(
            pytree_transpose, self.in_structure(), self.pattern.transpose()
        )

    def in_structure(self):
        return self.input_structure

    def out_structure(self):
        return self.output_structure


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
    fn: Callable
    x: PyTree[Array]
    args: Optional[PyTree[Any]]
    pattern: Pattern = field()  # overwrite abstract field

    def __init__(
        self,
        fn: Callable,
        x: PyTree[Array],
        args: Optional[PyTree[Array]] = None,
        pattern: Pattern = Pattern(),
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
        x = jtu.tree_map(jnp.asarray, x)
        fn = eqx.filter_closure_convert(fn, x, args)
        self.fn = fn
        self.x = x
        self.args = args
        self.pattern = pattern

    def mv(self, vector):
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        _, out = jax.jvp(fn, (self.x,), (vector,))
        return out

    def as_matrix(self):
        return self.materialise().as_matrix()

    def materialise(self):
        fn = _NoAuxIn(self.fn, self.args)
        jac, aux = jacobian(fn, self.in_size(), self.out_size(), has_aux=True)(self.x)
        out = PyTreeLinearOperator(jac, self.out_structure(), self.pattern)
        return _AuxLinearOperator(out, aux)

    def linearise(self):
        fn = _NoAuxIn(self.fn, self.args)
        (_, aux), lin = jax.linearize(fn, self.x)
        lin = _NoAuxOut(lin)
        out = FunctionLinearOperator(lin, self.in_structure(), self.pattern)
        return _AuxLinearOperator(out, aux)

    def transpose(self):
        if self.pattern.symmetric:
            return self
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        # Works because vjpfn is a PyTree
        _, vjpfn = jax.vjp(fn, self.x)
        vjpfn = _Unwrap(vjpfn)
        return FunctionLinearOperator(
            vjpfn, self.out_structure(), self.pattern.transpose()
        )

    def in_structure(self):
        return jax.eval_shape(lambda: self.x)

    def out_structure(self):
        fn = _NoAuxOut(_NoAuxIn(self.fn, self.args))
        return cached_eval_shape(fn, self.x)


# `input_structure` must be static as with `JacobianLinearOperator`
class FunctionLinearOperator(AbstractLinearOperator):
    fn: Callable[[PyTree[Array]], PyTree[Array]]
    input_structure: PyTree[jax.ShapeDtypeStruct] = eqx.static_field()
    pattern: Pattern = field()  # overwrite abstract field

    def __init__(
        self,
        fn: Callable[[PyTree[Array]], PyTree[Array]],
        input_structure: PyTree[jax.ShapeDtypeStruct],
        pattern: Pattern = Pattern(),
    ):
        # See matching comment in JacobianLinearOperator.
        fn = eqx.filter_closure_convert(fn, input_structure)
        self.fn = fn
        self.input_structure = input_structure
        self.pattern = pattern

    def mv(self, vector):
        return self.fn(vector)

    def as_matrix(self):
        return self.materialise().as_matrix()

    def materialise(self):
        # TODO(kidger): implement more efficiently, without the relinearisation
        zeros = jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), self.in_structure())
        jac = jacobian(self.fn, self.in_size(), self.out_size())(zeros)
        return PyTreeLinearOperator(jac, self.out_structure(), self.pattern)

    def linearise(self):
        return self

    def transpose(self):
        if self.pattern.symmetric:
            return self
        transpose_fn = jax.linear_transpose(self.fn, self.in_structure())
        # Works because transpose_fn is a PyTree
        return FunctionLinearOperator(
            transpose_fn, self.out_structure(), self.pattern.transpose()
        )

    def in_structure(self):
        return self.input_structure

    def out_structure(self):
        return cached_eval_shape(self.fn, self.in_structure())


# `structure` must be static as with `JacobianLinearOperator`
class IdentityLinearOperator(AbstractLinearOperator):
    structure: PyTree[jax.ShapeDtypeStruct] = eqx.static_field()

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

    def linearise(self):
        return self

    def in_structure(self):
        return self.structure

    def out_structure(self):
        return self.structure

    @property
    def pattern(self):
        return Pattern(
            symmetric=True,
            unit_diagonal=True,
            diagonal=True,
            positive_semidefinite=True,
        )


#
# Everything below here is private to Optimistix
#


class TangentLinearOperator(AbstractLinearOperator):
    primal: AbstractLinearOperator
    tangent: AbstractLinearOperator

    def __post_init__(self):
        assert self.primal.in_structure() == self.tangent.in_structure()
        assert self.primal.out_structure() == self.tangent.out_structure()
        assert self.primal.pattern == self.tangent.pattern

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
        primal_out, tangent_out = eqx.filter_jvp(materialise, self.primal, self.tangent)
        return TangentLinearOperator(primal_out, tangent_out)

    def linearise(self):
        linearise = lambda operator: operator.linearise()
        primal_out, tangent_out = eqx.filter_jvp(linearise, self.primal, self.tangent)
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


class _AddLinearOperator(AbstractLinearOperator):
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

    def materialise(self):
        return self.operator1.materialise() + self.operator2.materialise()

    def linearise(self):
        return self.operator1.linearise() + self.operator2.linearise()

    def transpose(self):
        return self.operator1.transpose() + self.operator2.transpose()

    def in_structure(self):
        return self.operator1.in_structure()

    def out_structure(self):
        return self.operator1.out_structure()

    @property
    def pattern(self):
        pattern1 = self.operator1.pattern
        pattern2 = self.operator2.pattern
        symmetric = pattern1.symmetric and pattern2.symmetric
        unit_diagonal = False
        lower_triangular = pattern1.lower_triangular and pattern2.lower_triangular
        upper_triangular = pattern1.upper_triangular and pattern2.upper_triangular
        diagonal = pattern1.diagonal and pattern2.diagonal
        positive_semidefinite = (
            pattern1.positive_semidefinite and pattern2.positive_semidefinite
        )
        negative_semidefinite = (
            pattern1.negative_semidefinite and pattern2.negative_semidefinite
        )
        nonsingular = False  # default
        return Pattern(
            symmetric=symmetric,
            unit_diagonal=unit_diagonal,
            lower_triangular=lower_triangular,
            upper_triangular=upper_triangular,
            diagonal=diagonal,
            positive_semidefinite=positive_semidefinite,
            negative_semidefinite=negative_semidefinite,
            nonsingular=nonsingular,
        )


class _MulLinearOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    scalar: Scalar

    def mv(self, vector):
        return (self.scalar * self.operator.mv(vector) ** ω).ω

    def as_matrix(self):
        return self.scalar * self.operator.as_matrix()

    def materialise(self):
        return self.scalar * self.operator.materialise()

    def linearise(self):
        return self.scalar * self.operator.linearise()

    def transpose(self):
        return self.scalar * self.operator.transpose()

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()

    @property
    def pattern(self):
        return self.operator.pattern


class _ComposedLinearOperator(AbstractLinearOperator):
    operator1: AbstractLinearOperator
    operator2: AbstractLinearOperator

    def __post_init__(self):
        if self.operator2.in_structure() != self.operator1.out_structure():
            raise ValueError("Incompatible linear operator structures")

    def mv(self, vector):
        return self.operator2.mv(self.operator1.mv(vector))

    def as_matrix(self):
        return jnp.matmul(
            self.operator2.as_matrix(),
            self.operator1.as_matrix(),
            precision=lax.Precision.HIGHEST,
        )

    def materialise(self):
        return self.operator2.materialise() @ self.operator1.materialise()

    def linearise(self):
        return self.operator2.linearise() @ self.operator1.linearise()

    def transpose(self):
        return self.operator1.transpose() @ self.operator2.transpose()

    def in_structure(self):
        return self.operator1.in_structure()

    def out_structure(self):
        return self.operator2.out_structure()

    @property
    def pattern(self):
        pattern1 = self.operator1.pattern
        pattern2 = self.operator2.pattern
        symmetric = pattern1.symmetric and pattern2.symmetric
        lower_triangular = pattern1.lower_triangular and pattern2.lower_triangular
        upper_triangular = pattern1.upper_triangular and pattern2.upper_triangular
        diagonal = pattern1.diagonal and pattern2.diagonal
        if lower_triangular or upper_triangular or diagonal:
            unit_diagonal = pattern1.unit_diagonal and pattern2.unit_diagonal
        else:
            unit_diagonal = False
        positive_semidefinite = (
            pattern1.positive_semidefinite and pattern2.positive_semidefinite
        )
        negative_semidefinite = (
            pattern1.negative_semidefinite and pattern2.negative_semidefinite
        )
        nonsingular = pattern1.nonsingular and pattern2.nonsingular
        return Pattern(
            symmetric=symmetric,
            unit_diagonal=unit_diagonal,
            lower_triangular=lower_triangular,
            upper_triangular=upper_triangular,
            diagonal=diagonal,
            positive_semidefinite=positive_semidefinite,
            negative_semidefinite=negative_semidefinite,
            nonsingular=nonsingular,
        )


class _AuxLinearOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    aux: PyTree[Array]

    def mv(self, vector):
        return self.operator.mv(vector)

    def as_matrix(self):
        return self.operator.as_matrix()

    def materialise(self):
        return self.operator.materialise()

    def linearise(self):
        return self.operator.linearise()

    def transpose(self):
        return self.operator.transpose()

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()

    @property
    def pattern(self):
        return self.operator.pattern
