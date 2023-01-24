import abc
import functools as ft
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.interpreters.ad as ad
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree, Shaped

from .custom_types import sentinel
from .linear_operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    TangentLinearOperator,
)
from .solution import RESULTS, Solution


#
# _linear_solve_p
#


def _to_shapedarray(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.core.ShapedArray(x.shape, x.dtype)
    else:
        return x


def _to_struct(x):
    if isinstance(x, jax.core.ShapedArray):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)
    elif isinstance(x, jax.core.AbstractValue):
        raise NotImplementedError(
            "`optimistic.linear_solve` only supports working with JAX arrays; not "
            "other abstract values. Got {type(x)}."
        )
    else:
        return x


def _assert_none(x):
    assert x is None


def _is_none(x):
    return x is None


def _sum(*args):
    return sum(args)


def _call(state, vector, options, solver):
    return solver.compute(state, vector, options)


@eqxi.filter_primitive_def
def _linear_solve_impl(_, state, vector, options, solver):
    out = _call(state, vector, options, solver)
    return eqxi.nontraceable(
        out, name="optimistix.linear_solve with respect to a closed-over value"
    )


@eqxi.filter_primitive_def
def _linear_solve_abstract_eval(_, state, vector, options, solver):
    state, vector, options, solver = jtu.tree_map(
        _to_struct, (state, vector, options, solver)
    )
    out = eqx.filter_eval_shape(_call, state, vector, options, solver)
    out = jtu.tree_map(_to_shapedarray, out)
    return out


@eqxi.filter_primitive_jvp
def _linear_solve_jvp(primals, tangents):
    operator, state, vector, options, solver = primals
    t_operator, t_state, t_vector, t_options, t_solver = tangents
    jtu.tree_map(_assert_none, (t_state, t_options, t_solver))
    del t_state, t_options, t_solver

    solution, result, stats = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver
    )

    #
    # Consider the primal problem of linearly solving for x in Ax=b.
    # Let ^ denote pseudoinverses, ᵀ denote transposes, and ' denote tangents.
    # The linear_solve routine returns specifically the pseudoinverse solution, i.e.
    # x = A^b
    # Therefore x' = A^'b + A^b'
    #
    # Now A^' = -A^A'A^ + A^A^ᵀAᵀ'(I - AA^) + (I - A^A)Aᵀ'A^ᵀA^
    #
    # (Source: https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Derivative)
    #
    # Noting that one of these terms cancels as (I - AA^)b = 0, we have that
    # x' = -A^A'x + (I - A^A)Aᵀ'A^ᵀx + A^b'
    # which may be calculated as
    #
    # x' = A^(-A'x - Ay + b') + y
    # where
    # y = Aᵀ'A^ᵀx
    #
    # and when A is nonsingular note that this reduces to just
    #
    # x' = A^(-A'x + b')
    #
    vecs = []
    sols = []
    if any(t is not None for t in jtu.tree_leaves(t_vector, is_leaf=_is_none)):
        vecs.append(
            jtu.tree_map(eqxi.materialise_zeros, vector, t_vector, is_leaf=_is_none)
        )
    if any(t is not None for t in jtu.tree_leaves(t_operator, is_leaf=_is_none)):
        t_operator = TangentLinearOperator(operator, t_operator)
        t_operator = t_operator.linearise()  # optimise for matvecs
        vec = (-t_operator.mv(solution) ** ω).ω
        vecs.append(vec)
        if solver.is_maybe_singular():
            operator_transpose = operator.transpose()
            state_transpose, options_transpose = solver.transpose(state, options)
            tmp1, _, _ = eqxi.filter_primitive_bind(
                linear_solve_p,
                operator_transpose,
                state_transpose,
                solution,
                options_transpose,
                solver,
            )
            tmp2 = t_operator.transpose().mv(tmp1)
            tmp3 = operator.mv(tmp2)
            tmp4 = (-(tmp3**ω)).ω
            vecs.append(tmp4)
            sols.append(tmp2)
    vecs = jtu.tree_map(_sum, *vecs)
    sol, _, _ = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vecs, options, solver
    )
    sols.append(sol)
    t_solution = jtu.tree_map(_sum, *sols)

    out = solution, result, stats
    t_out = t_solution, None, jtu.tree_map(lambda _: None, stats)
    return out, t_out


def _is_undefined(x):
    return isinstance(x, ad.UndefinedPrimal)


def _assert_defined(x):
    assert not _is_undefined(x)


def _keep_undefined(v, ct):
    if _is_undefined(v):
        return ct
    else:
        return None


@eqxi.filter_primitive_transpose
def _linear_solve_transpose(inputs, cts_out):
    cts_solution, _, _ = cts_out
    operator, state, vector, options, solver = inputs
    jtu.tree_map(
        _assert_defined, (operator, state, options, solver), is_leaf=_is_undefined
    )
    operator_transpose = operator.transpose()
    state_transpose, options_transpose = solver.transpose(state, options)
    cts_vector, _, _ = eqxi.filter_primitive_bind(
        linear_solve_p,
        operator_transpose,
        state_transpose,
        cts_solution,
        options_transpose,
        solver,
    )
    cts_vector = jtu.tree_map(
        _keep_undefined, vector, cts_vector, is_leaf=_is_undefined
    )
    operator_none = jtu.tree_map(lambda _: None, operator)
    state_none = jtu.tree_map(lambda _: None, state)
    options_none = jtu.tree_map(lambda _: None, options)
    solver_none = jtu.tree_map(lambda _: None, solver)
    return operator_none, state_none, cts_vector, options_none, solver_none


linear_solve_p = eqxi.create_vprim(
    "linear_solve",
    _linear_solve_impl,
    _linear_solve_abstract_eval,
    _linear_solve_jvp,
    _linear_solve_transpose,
)
eqxi.register_impl_finalisation(linear_solve_p)


#
# linear_solve
#


_SolverState = TypeVar("_SolverState")


class AbstractLinearSolver(eqx.Module):
    @abc.abstractmethod
    def is_maybe_singular(self):
        ...

    @abc.abstractmethod
    def init(
        self, operator: AbstractLinearOperator, options: Dict[str, Any]
    ) -> _SolverState:
        ...

    @abc.abstractmethod
    def compute(
        self, state: _SolverState, vector: Shaped[Array, " b"], options: Dict[str, Any]
    ) -> Tuple[Shaped[Array, " a"], RESULTS, Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def transpose(
        self, state: _SolverState, options: Dict[str, Any]
    ) -> Tuple[_SolverState, Dict[str, Any]]:
        ...


_qr_token = eqxi.str2jax("QR")
_diagonal_token = eqxi.str2jax("Diagonal")
_triangular_token = eqxi.str2jax("Triangular")
_cg_token = eqxi.str2jax("CG")
_cholesky_token = eqxi.str2jax("Cholesky")
_lu_token = eqxi.str2jax("LU")


# Ugly delayed import because we have the dependency chain
# linear_solve -> AutoLinearSolver -> {Cholesky,...} -> AbstractLinearSolver
# but we want linear_solver and AbstractLinearSolver in the same file.
def _lookup(token) -> Callable[[], AbstractLinearSolver]:
    from . import solver

    if jax.config.jax_enable_x64:
        tol = 1e-12
    else:
        tol = 1e-6

    # We set CG(materialise=True) because CG evalutes `operator.mv` twice. By
    # materialising the matrix we ensure that we don't silently double compile time.
    # (In contrast the out-of-memory issue that materialisation might get induce is a
    # loud error.)
    _lookup_dict = {
        _qr_token: solver.QR,
        _diagonal_token: solver.Diagonal,
        _triangular_token: solver.Triangular,
        _cg_token: ft.partial(solver.CG, rtol=tol, atol=tol, materialise=True),
        _cholesky_token: solver.Cholesky,
        _lu_token: solver.LU,
    }
    return _lookup_dict[token]


class AutoLinearSolver(AbstractLinearSolver):
    maybe_singular: bool = False

    def is_maybe_singular(self):
        return self.maybe_singular

    def _auto_select_solver(self, operator: AbstractLinearOperator):
        # Of these solvers: QR, Diagonal, Triangular, CG can handle ill-posed
        # problems, i.e. they converge to the pseudoinverse solution.
        if operator.in_size() != operator.out_size():
            token = _qr_token
        elif operator.pattern.diagonal:
            token = _diagonal_token
        elif operator.pattern.lower_triangular or operator.pattern.upper_triangular:
            token = _triangular_token
        elif operator.pattern.symmetric:
            token = _cg_token
        else:
            if self.maybe_singular:
                token = _qr_token
            else:
                token = _lu_token
        return token

    def init(self, operator, options):
        token = self._auto_select_solver(operator)
        return token, _lookup(token)().init(operator, options)

    def compute(self, state, vector, options):
        token, state = state
        solver = _lookup(token)()
        solution, result, _ = solver.compute(state, vector, options)
        return solution, result, {}

    def transpose(self, state, options):
        token, state = state
        solver = _lookup(token)()
        return solver.transpose(state, options)


# TODO(kidger): gmres, bicgstab
# TODO(kidger): support auxiliary outputs
@eqx.filter_jit
def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Array],
    solver: AbstractLinearSolver = AutoLinearSolver(),
    options: Optional[Dict[str, Any]] = None,
    *,
    state: PyTree[Array] = sentinel,
    throw: bool = True,
) -> Solution:
    if options is None:
        options = {}
    if jax.eval_shape(lambda: vector) != operator.out_structure():
        raise ValueError("Vector and operator structures do not match")
    if isinstance(operator, IdentityLinearOperator):
        return Solution(
            value=vector, result=RESULTS.successful, state=state, stats={}, aux=None
        )
    if state == sentinel:
        state = solver.init(operator, options)
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        dynamic_state = lax.stop_gradient(dynamic_state)
        state = eqx.combine(dynamic_state, static_state)

    state, options, solver = eqxi.nondifferentiable((state, options, solver))
    solution, result, stats = eqxi.filter_primitive_bind(
        linear_solve_p, operator, state, vector, options, solver
    )
    # TODO: prevent forward-mode autodiff through stats
    stats = eqxi.nondifferentiable_backward(stats)

    has_nans = jnp.any(
        jnp.stack([jnp.any(jnp.isnan(x)) for x in jtu.tree_leaves(solution)])
    )
    result = jnp.where(
        (result == RESULTS.successful) & has_nans, RESULTS.linear_singular, result
    )
    sol = Solution(value=solution, result=result, state=state, stats=stats, aux=None)

    error_index = eqxi.unvmap_max(result)
    sol = eqxi.branched_error_if(
        sol,
        throw & (result != RESULTS.successful),
        error_index,
        RESULTS.reverse_lookup,
    )
    return sol
