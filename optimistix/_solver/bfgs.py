from collections.abc import Callable
from typing import Any, Generic, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import (
    Aux,
    Constraint,
    DescentState,
    EqualityOut,
    Fn,
    InequalityOut,
    SearchState,
    Y,
)
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    evaluate_constraint,
    filter_cond,
    lin_to_grad,
    max_norm,
    tree_clip,
    tree_dot,
    tree_full_like,
    verbose_print,
)
from .._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .cauchy_point import CauchyNewtonDescent
from .gauss_newton import NewtonDescent


def identity_pytree(pytree: PyTree[Array]) -> lx.PyTreeLinearOperator:
    """Create an identity pytree `I` such that
    `pytree = lx.PyTreeLinearOperator(I).mv(pytree)`

    **Arguments**:

    - `pytree`: A pytree such that the output of `_identity_pytree` is the identity
        with respect to pytrees of the same shape as `pytree`.

    **Returns**:

    A `lx.PyTreeLinearOperator` with input and output shape the shape of `pytree`.
    """
    leaves, structure = jtu.tree_flatten(pytree)
    eye_structure = structure.compose(structure)
    eye_leaves = []
    for i1, l1 in enumerate(leaves):
        for i2, l2 in enumerate(leaves):
            if i1 == i2:
                eye_leaves.append(
                    jnp.eye(jnp.size(l1)).reshape(jnp.shape(l1) + jnp.shape(l2))
                )
            else:
                eye_leaves.append(jnp.zeros(jnp.shape(l1) + jnp.shape(l2)))

    # This has a Lineax positive_semidefinite tag. This is okay because the BFGS update
    # preserves positive-definiteness.
    return lx.PyTreeLinearOperator(
        jtu.tree_unflatten(eye_structure, eye_leaves),
        jax.eval_shape(lambda: pytree),
        lx.positive_semidefinite_tag,
    )


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


def bfgs_update(
    f_eval,
    grad,
    prev_grad,
    hessian,
    hessian_inv,
    y_eval,
    y_diff,
    bounds,
    constraint_residual,
    constraint_bound,
    constraint_jac,
):
    grad_diff = (grad**ω - prev_grad**ω).ω
    inner = tree_dot(grad_diff, y_diff)

    def update(hessian, hessian_inv):
        if hessian is None:
            assert hessian_inv is not None
            # Use Woodbury identity for rank-1 update of approximate Hessian.
            inv_mvp = hessian_inv.mv(grad_diff)
            mvp_inner = tree_dot(grad_diff, inv_mvp)
            diff_outer = _outer(y_diff, y_diff)
            mvp_outer = _outer(y_diff, inv_mvp)
            term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
            term2 = ((_outer(inv_mvp, y_diff) ** ω + mvp_outer**ω) / inner).ω
            new_hessian_inv = lx.PyTreeLinearOperator(
                (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,
                output_structure=jax.eval_shape(lambda: prev_grad),
                tags=lx.positive_semidefinite_tag,
            )
            return None, new_hessian_inv
        else:
            assert hessian_inv is None
            # BFGS update to the operator directly
            mvp = hessian.mv(y_diff)
            term1 = (_outer(grad_diff, grad_diff) ** ω / inner).ω
            term2 = (_outer(mvp, mvp) ** ω / tree_dot(y_diff, mvp)).ω
            hessian = lx.PyTreeLinearOperator(
                (hessian.pytree**ω + term1**ω - term2**ω).ω,
                output_structure=jax.eval_shape(lambda: prev_grad),
                tags=lx.positive_semidefinite_tag,
            )
            return hessian, None

    def no_update(hessian, hessian_inv):
        return hessian, hessian_inv

    # In particular inner = 0 on the first step (as then state.grad=0), and so for
    # this we jump straight to the line search.
    # Likewise we get inner <= eps on convergence, and so again we make no update
    # to avoid a division by zero.
    inner_nonzero = inner > jnp.finfo(inner.dtype).eps
    hessian, hessian_inv = filter_cond(
        inner_nonzero, update, no_update, hessian, hessian_inv
    )
    if hessian is None:
        return FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian_inv)
    else:
        return FunctionInfo.EvalGradHessian(
            f_eval,
            grad,
            hessian,
            y_eval,
            bounds,
            constraint_residual,
            constraint_bound,
            constraint_jac,
        )


_Hessian = TypeVar(
    "_Hessian", FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv
)


class _OldBFGSState(
    eqx.Module, Generic[Y, Aux, SearchState, DescentState, _Hessian], strict=True
):
    # Updated every search step
    first_step: Bool[Array, ""]
    y_eval: Y
    search_state: SearchState
    # Updated after each descent step
    f_info: _Hessian
    aux: Aux
    descent_state: DescentState
    # Used for termination
    terminate: Bool[Array, ""]
    result: RESULTS
    # Used in compat.py
    num_accepted_steps: Int[Array, ""]


class AbstractOldBFGS(
    AbstractMinimiser[Y, Aux, _OldBFGSState], Generic[Y, Aux, _Hessian], strict=True
):
    """Abstract BFGS (Broyden--Fletcher--Goldfarb--Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.

    This abstract version may be subclassed to choose alternative descent and searches.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    - `clip`: A boolean indicating whether to clip the solution to the bounds after each
        step. If `True`, then bounds must be passed in the `bounds` argument.
        Defaults to `False`.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    use_inverse: AbstractVar[bool]
    descent: AbstractVar[AbstractDescent[Y, _Hessian, Any]]
    search: AbstractVar[AbstractSearch[Y, _Hessian, FunctionInfo.Eval, Any]]
    verbose: AbstractVar[frozenset[str]]

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _OldBFGSState:
        clip = options.get("clip", False)
        if clip and bounds is None:
            raise ValueError("If clip is True, bounds must be provided.")

        if constraint is not None:
            evaluated = evaluate_constraint(constraint, y)
            constraint_residual, constraint_bound, constraint_jacobians = evaluated
        else:
            constraint_residual = constraint_bound = constraint_jacobians = None

        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        if self.use_inverse:
            hessian_inv = identity_pytree(y)
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, hessian_inv)
        else:
            hessian = identity_pytree(y)
            f_info = FunctionInfo.EvalGradHessian(
                f,
                grad,
                hessian,
                y,
                bounds,
                constraint_residual,
                constraint_bound,
                constraint_jacobians,  # pyright: ignore  # TODO fix this!!
            )
        f_info_struct = eqx.filter_eval_shape(lambda: f_info)
        return _OldBFGSState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _OldBFGSState,
        tags: frozenset[object],
    ) -> tuple[Y, _OldBFGSState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        clip = options.get("clip", False)

        if constraint is not None:
            evaluated = evaluate_constraint(constraint, state.y_eval)
            constraint_residual, constraint_bound, constraint_jacobians = evaluated
        else:
            constraint_residual = constraint_bound = constraint_jacobians = None

        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )

        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval, bounds, constraint_residual),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)

            y_diff = (state.y_eval**ω - y**ω).ω

            if self.use_inverse:
                hessian = None
                hessian_inv = state.f_info.hessian_inv
            else:
                hessian = state.f_info.hessian
                hessian_inv = None

            f_eval_info = bfgs_update(
                f_eval,
                grad,
                state.f_info.grad,
                hessian,
                hessian_inv,
                state.y_eval,
                y_diff,
                bounds,
                constraint_residual,
                constraint_bound,
                constraint_jacobians,
            )

            descent_state = self.descent.query(
                state.y_eval,
                f_eval_info,  # pyright: ignore
                descent_state,
            )
            f_diff = f_eval - state.f_info.f
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = filter_cond(
            accept, accepted, rejected, state.descent_state
        )

        if len(self.verbose) > 0:
            verbose_loss = "loss" in self.verbose
            verbose_step_size = "step_size" in self.verbose
            verbose_y = "y" in self.verbose
            loss_eval = f_eval
            loss = state.f_info.f
            verbose_print(
                (verbose_loss, "Loss on this step", loss_eval),
                (verbose_loss, "Loss on the last accepted step", loss),
                (verbose_step_size, "Step size", step_size),
                (verbose_y, "y", state.y_eval),
                (verbose_y, "y on the last accepted step", y),
            )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        if clip and bounds is not None:
            y_eval = tree_clip(y_eval, *bounds)

        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        state = _OldBFGSState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
        )
        return y, state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _OldBFGSState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        constraint: Union[Constraint[Y, EqualityOut, InequalityOut], None],
        bounds: Union[tuple[Y, Y], None],
        state: _OldBFGSState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


# TODO: retire AbstractBFGS and inherit from AbstractQuasiNewton instead. Now not doing
# that to avoid cluttering up QuasiNewton with support for constrained FunctionInfo
# stuff until we have settled on an API for that.
# (Several composed constrained optimisers that we test inherit from AbstractBFGS.)
class OldBFGS(AbstractOldBFGS[Y, Aux, _Hessian], strict=True):
    """BFGS (Broyden--Fletcher--Goldfarb--Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    - `clip`: A boolean indicating whether to clip the solution to the bounds after each
        step. If `True`, then bounds must be passed in the `bounds` argument.
        Defaults to `False`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: BacktrackingArmijo
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
        self.search = BacktrackingArmijo()
        self.verbose = verbose


OldBFGS.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: The BFGS algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves). The
    default is (b), denoted via `use_inverse=True`. Note that this is incompatible with
    line search methods like [`optimistix.ClassicalTrustRegion`][], which use the
    Hessian approximation `B` as part of their own computations.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example 
    `verbose=frozenset({"step_size", "loss"})`.
"""


# TODO: inherit from AbstractQuasiNewton here
class BFGS_B(AbstractOldBFGS[Y, Aux, FunctionInfo.EvalGradHessian], strict=True):
    """Bounded BFGS (Broyden--Fletcher--Goldfarb--Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information. This version also supports box constraints on the variables, which
    are handled via a box-projected Cauchy-Newton step: a minimiser is identified along
    the piecewise-linear projected gradient direction, and then used to define the
    active bound constraints while optimising in the unconstrained variables. The
    advantage over a simple box clip on the computed step is that trade-offs between
    the optimisation variables are taken into account when computing the constrained
    step.
    Does not support the use of an inverse Hessian approximation.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.

    (Note that while clipping is technically possible due to its support in
    [`optimistix.AbstractBFGS`][], it is redundant here and hence not listed.)
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: CauchyNewtonDescent
    search: BacktrackingArmijo
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = False
        self.descent = CauchyNewtonDescent(linear_solver=lx.SVD())
        # TODO(jhaffner): replace the linear solver with a more efficient one.
        self.search = BacktrackingArmijo()
        self.verbose = verbose


BFGS_B.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example 
    `verbose=frozenset({"step_size", "loss"})`.
"""
