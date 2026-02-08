import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Aux, DescentState, Fn, HessianUpdateState, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    filter_cond,
    lin_to_grad,
    max_norm,
    tree_dot,
    tree_full_like,
    tree_where,
    verbose_print,
)
from .._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from .._solution import RESULTS
from .backtracking import BacktrackingArmijo
from .gauss_newton import NewtonDescent


_Hessian = TypeVar(
    "_Hessian", FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv
)


def _identity_pytree(pytree: PyTree[Array]) -> lx.PyTreeLinearOperator:
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


class _QuasiNewtonState(
    eqx.Module,
    Generic[Y, Aux, SearchState, DescentState, _Hessian, HessianUpdateState],
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
    # update state
    hessian_update_state: HessianUpdateState


class AbstractQuasiNewton(
    AbstractMinimiser[Y, Aux, _QuasiNewtonState],
    Generic[Y, Aux, _Hessian, HessianUpdateState],
):
    """Abstract quasi-Newton minimisation algorithm.

    Base class for quasi-Newton solvers, which create approximations to the Hessian or
    the inverse Hessian by accumulating gradient information over multiple iterations.
    Optimistix currently includes the following three variants:
    [`optimistix.BFGS`][], [`optimistix.DFP`][] and [`optimistix.LBFGS`][], each of
    which may be used to either approximate the Hessian or its inverse.
    The concrete classes may be subclassed to choose alternative descents and searches.

    Alternative flavors of quasi-Newton approximations may be implemented by subclassing
    `AbstractQuasiNewton` and providing implementations for the abstract methods
    `init_hessian` and `update_hessian`. The former is called to initialize the Hessian
    structure and the Hessian update state, while the latter is called to compute an
    update to the approximation of the Hessian or the inverse Hessian.

    Already supported schemes to form inverse Hessian and Hessian approximations are
    implemented in `optimistix.AbstractBFGS`, `optimistix.AbstractDFP` and
    `optimistix.AbstractLBFGS`.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    use_inverse: AbstractVar[bool]
    descent: AbstractVar[AbstractDescent[Y, _Hessian, Any]]
    search: AbstractVar[AbstractSearch[Y, _Hessian, FunctionInfo.Eval, Any]]
    verbose: AbstractVar[frozenset[str]]

    @abc.abstractmethod
    def init_hessian(
        self, y: Y, f: Scalar, grad: Y
    ) -> tuple[_Hessian, HessianUpdateState]:
        """Initialize the Hessian structure and Hessian update state.

        Set up a template structure of the Hessian to be used (with dummy values), as
        well as the state of the update method, which can be used to store past
        gradients for limited-memory Hessian approximations.
        """

    @abc.abstractmethod
    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: HessianUpdateState,
    ) -> tuple[_Hessian, HessianUpdateState]:
        """Update the Hessian approximation.

        This is called in the `step` method to update the Hessian approximation based on
        the current and previous iterates, their gradients, and the previous Hessian,
        whenever a step has been accepted and we query the descent for a new direction.

        Implementations should provide an update for the Hessian approximation or its
        inverse, and toggle updates as appropriate to maintain positive-definiteness
        of the operator.
        """

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _QuasiNewtonState:
        f = tree_full_like(f_struct, 0)
        grad = tree_full_like(y, 0)
        f_info, hessian_update_state = self.init_hessian(y, f, grad)
        f_info_struct = eqx.filter_eval_shape(lambda: f_info)

        return _QuasiNewtonState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            descent_state=self.descent.init(y, f_info_struct),
            terminate=jnp.array(False),
            result=RESULTS.successful,
            num_accepted_steps=jnp.array(0),
            hessian_update_state=hessian_update_state,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
        tags: frozenset[object],
    ) -> tuple[Y, _QuasiNewtonState, Aux]:
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, lin_fn, aux_eval = jax.linearize(
            lambda _y: fn(_y, args), state.y_eval, has_aux=True
        )
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode=autodiff_mode)

            f_eval_info, hessian_update_state = self.update_hessian(
                y,
                state.y_eval,
                state.f_info,
                FunctionInfo.EvalGrad(f_eval, grad),
                state.hessian_update_state,
            )

            descent_state = self.descent.query(
                state.y_eval,
                f_eval_info,  # pyright: ignore
                descent_state,
            )
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return (
                state.y_eval,
                f_eval_info,
                aux_eval,
                descent_state,
                terminate,
                hessian_update_state,
            )

        def rejected(descent_state):
            return (
                y,
                state.f_info,
                state.aux,
                descent_state,
                jnp.array(False),
                state.hessian_update_state,
            )

        y, f_info, aux, descent_state, terminate, hessian_update_state = filter_cond(
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
        result = RESULTS.where(
            search_result == RESULTS.successful, descent_result, search_result
        )

        prev_aux = tree_where(state.first_step, aux, state.aux)
        state = _QuasiNewtonState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
            num_accepted_steps=state.num_accepted_steps + jnp.where(accept, 1, 0),
            hessian_update_state=hessian_update_state,
        )
        return y, state, prev_aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _QuasiNewtonState,
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
        state: _QuasiNewtonState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class AbstractBFGS(AbstractQuasiNewton[Y, Aux, _Hessian, None]):
    """Abstract version of the BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimisation
    algorithm. This class may be subclassed to implement custom solvers with alternative
    searches and descent methods that use the BFGS update to approximate the Hessian or
    the inverse Hessian.
    """

    def init_hessian(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, None]:
        identity_operator = _identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_operator)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_operator)
        return f_info, None  # pyright: ignore

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: None,
    ) -> tuple[_Hessian, None]:
        f_eval = f_eval_info.f
        grad = f_eval_info.grad
        y_diff = (y_eval**ω - y**ω).ω
        grad_diff = (grad**ω - f_info.grad**ω).ω
        inner = tree_dot(grad_diff, y_diff)

        # In particular inner = 0 on the first step (as then state.grad=0), and so for
        # this we jump straight to the line search.
        # Likewise we get inner <= eps on convergence, and so again we make no update
        # to avoid a division by zero.
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps

        def no_update(args):
            *_, f_info = args
            if self.use_inverse:
                return f_info.hessian_inv
            else:
                return f_info.hessian

        def update(args):
            inner, grad_diff, y_diff, f_info = args
            if self.use_inverse:
                assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
                hessian_inv = f_info.hessian_inv
                # Use Woodbury identity for rank-1 update of approximate Hessian.
                inv_mvp = hessian_inv.mv(grad_diff)
                mvp_inner = tree_dot(grad_diff, inv_mvp)
                diff_outer = _outer(y_diff, y_diff)
                mvp_outer = _outer(y_diff, inv_mvp)
                term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
                term2 = ((_outer(inv_mvp, y_diff) ** ω + mvp_outer**ω) / inner).ω
                new_hessian_inv = lx.PyTreeLinearOperator(
                    (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian_inv
            else:
                assert isinstance(f_info, FunctionInfo.EvalGradHessian)
                hessian = f_info.hessian
                # BFGS update to the operator directly
                mvp = hessian.mv(y_diff)
                term1 = (_outer(grad_diff, grad_diff) ** ω / inner).ω
                term2 = (_outer(mvp, mvp) ** ω / tree_dot(y_diff, mvp)).ω
                new_hessian = lx.PyTreeLinearOperator(
                    (hessian.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian

        args = (inner, grad_diff, y_diff, f_info)
        hessian = filter_cond(
            inner_nonzero,
            update,
            no_update,
            args,
        )

        # We're using pyright: ignore here because the type of `FunctionInfo` depends on
        # the `use_inverse` attribute.
        # https://github.com/patrick-kidger/optimistix/pull/135#discussion_r2155452558
        if self.use_inverse:
            return FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian), None  # pyright: ignore
        else:
            return FunctionInfo.EvalGradHessian(f_eval, grad, hessian), None  # pyright: ignore


class BFGS(AbstractBFGS[Y, Aux, _Hessian]):
    """BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information. Uses the Broyden-Fletcher-Goldfarb-Shanno formula to compute the
    updates to the Hessian and or to the Hessian inverse.
    See [https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm).

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
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


BFGS.__init__.__doc__ = """**Arguments:**

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
    searches like [`optimistix.ClassicalTrustRegion`][], which use the Hessian 
    approximation `B` as part of their computations.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example
    `verbose=frozenset({"step_size", "loss"})`.
"""


class AbstractDFP(AbstractQuasiNewton[Y, Aux, _Hessian, None]):
    """Abstract version of the DFP (Davidon–Fletcher–Powell) minimisation algorithm.
    This class may be subclassed to implement custom solvers with alternative searches
    and descent methods that use the DFP update to approximate the Hessian or the
    inverse Hessian.
    """

    def init_hessian(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, None]:
        identity_operator = _identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_operator)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_operator)
        return f_info, None  # pyright: ignore

    def update_hessian(
        self,
        y: Y,
        y_eval: Y,
        f_info: _Hessian,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: None,
    ) -> tuple[_Hessian, None]:
        f_eval = f_eval_info.f
        grad = f_eval_info.grad
        y_diff = (y_eval**ω - y**ω).ω
        grad_diff = (grad**ω - f_info.grad**ω).ω
        inner = tree_dot(grad_diff, y_diff)

        # In particular inner = 0 on the first step (as then state.grad=0), and so for
        # this we jump straight to the line search.
        # Likewise we get inner <= eps on convergence, and so again we make no update
        # to avoid a division by zero.
        inner_nonzero = inner > jnp.finfo(inner.dtype).eps

        def no_update(args):
            *_, f_info = args
            if self.use_inverse:
                return f_info.hessian_inv
            else:
                return f_info.hessian

        def update(args):
            inner, grad_diff, y_diff, f_info = args
            if self.use_inverse:
                assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
                hessian_inv = f_info.hessian_inv
                inv_mvp = hessian_inv.mv(grad_diff)
                term1 = (_outer(y_diff, y_diff) ** ω / inner).ω
                term2 = (_outer(inv_mvp, inv_mvp) ** ω / tree_dot(grad_diff, inv_mvp)).ω
                new_hessian_inv = lx.PyTreeLinearOperator(
                    (hessian_inv.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian_inv
            else:
                assert isinstance(f_info, FunctionInfo.EvalGradHessian)
                hessian = f_info.hessian
                mvp = hessian.mv(y_diff)
                mvp_inner = tree_dot(y_diff, mvp)
                diff_outer = _outer(grad_diff, grad_diff)
                mvp_outer = _outer(grad_diff, mvp)
                term1 = (((inner + mvp_inner) * (diff_outer**ω)) / (inner**2)).ω
                term2 = ((_outer(mvp, grad_diff) ** ω + mvp_outer**ω) / inner).ω
                new_hessian = lx.PyTreeLinearOperator(
                    (hessian.pytree**ω + term1**ω - term2**ω).ω,  # pyright: ignore
                    output_structure=jax.eval_shape(lambda: grad_diff),
                    tags=lx.positive_semidefinite_tag,
                )
                return new_hessian

        args = (inner, grad_diff, y_diff, f_info)
        hessian = filter_cond(
            inner_nonzero,
            update,
            no_update,
            args,
        )

        # We're using pyright: ignore here because the type of `FunctionInfo` depends on
        # the `use_inverse` attribute.
        # https://github.com/patrick-kidger/optimistix/pull/135#discussion_r2155452558
        if self.use_inverse:
            return FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian), None  # pyright: ignore
        else:
            return FunctionInfo.EvalGradHessian(f_eval, grad, hessian), None  # pyright: ignore


class DFP(AbstractDFP[Y, Aux, _Hessian]):
    """DFP (Davidon–Fletcher–Powell) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information. Uses the Davidon-Fletcher-Powell formula to compute the updates to
    the Hessian and or to the Hessian inverse.
    See [https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula](https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula).

    [`optimistix.BFGS`][] is generally preferred, since it is more numerically stable on
    most problems.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
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


DFP.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `use_inverse`: The DFP algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves). The
    default is (b), denoted via `use_inverse=True`. Note that this is incompatible with
    searches like [`optimistix.ClassicalTrustRegion`][], which use the Hessian 
    approximation `B` as part of their computations.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example
    `verbose=frozenset({"step_size", "loss"})`.
"""
