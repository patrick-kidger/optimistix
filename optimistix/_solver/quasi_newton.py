import abc
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

from .._custom_types import Aux, DescentState, Fn, HessianUpdateState, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    filter_cond,
    lin_to_grad,
    max_norm,
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
from .gauss_newton import NewtonDescent


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


@jax.jit
def _lbfgs_operator_fn(
        pytree: PyTree[Array],
        y_diff_history: PyTree[Array],
        grad_diff_history: PyTree[Array],
        inner_history: Array,
        index_start: Array
):
    """
    LBFGS descent linear operator.

    """
    history_len = inner_history.shape[0]
    circ_index = (jnp.arange(history_len) + index_start) % history_len

    # First loop: iterate backwards and compute alpha coefficients
    def backward_iter(descent_direction, indx):
        y_diff, grad_diff, inner = jtu.tree_map(
            lambda x: x[indx], (y_diff_history, grad_diff_history, inner_history)
        )
        alpha = inner * tree_dot(y_diff, descent_direction)
        descent_direction = (descent_direction ** ω - alpha * grad_diff ** ω).ω
        return descent_direction, alpha

    # Second loop: iterate forwards and apply correction using stored alpha
    def forward_iter(args, indx):
        descent_direction, alpha = args
        current_alpha = alpha[indx]
        inner = inner_history[circ_index[indx]]
        y_diff, grad_diff = jtu.tree_map(
            lambda x: x[circ_index[indx]],
            (y_diff_history, grad_diff_history)
        )
        current_beta = inner * tree_dot(grad_diff, descent_direction)
        descent_direction = (
                descent_direction ** ω + (y_diff ** ω * (current_alpha - current_beta))
        ).ω
        return (descent_direction, alpha), None


    descent_direction, alpha = jax.lax.scan(
        backward_iter, pytree, circ_index, reverse=True
    )
    latest_y_diff, latest_grad_diff = jtu.tree_map(
        lambda x: x[index_start % history_len],
        (y_diff_history, grad_diff_history)
    )
    y_grad_diff_inner = tree_dot(latest_y_diff, latest_grad_diff)
    grad_diff_norm_sq = tree_dot(latest_grad_diff, latest_grad_diff)
    gamma_k = jnp.where(
        grad_diff_norm_sq > 1e-10, y_grad_diff_inner / grad_diff_norm_sq, 1.0
    )
    descent_direction = (gamma_k * descent_direction ** ω).ω
    (descent_direction, _), _ = jax.lax.scan(
        forward_iter, (descent_direction, alpha), jnp.arange(history_len), reverse=False
    )
    return descent_direction


def _make_lbfgs_operator(
        y_diff_history: PyTree[Array],
        grad_diff_history: PyTree[Array],
        inner_history: Array,
        index_start: Array,
):
    """Define a linear operator implementing the L-BFGS inverse Hessian approximation.

    This operator computes the action of the approximate inverse Hessian
    on a vector `pytree` using the limited-memory BFGS (L-BFGS) two-loop
    recursion. It does not materialize the matrix explicitly but returns a
    `lineax.FunctionLinearOperator`.

    - `y_diff_history`: History of parameter updates `s_k = x_{k+1} - x_k`
    - `grad_diff_history`: History of gradient updates `y_k = g_{k+1} - g_k`
    - `rho`: Reciprocal dot products `rho_k = 1 / ⟨s_k, y_k⟩`
    - `index_start`: Index of the most recent update in the circular buffer

    Returns a `lineax.FunctionLinearOperator` with input and output shape
    matching a single element of `residual_par`.

    """
    operator_func = eqx.Partial(
        _lbfgs_operator_fn,
        y_diff_history=y_diff_history,
        grad_diff_history=grad_diff_history,
        inner_history=inner_history,
        index_start=index_start
    )
    input_shape = jax.eval_shape(lambda: jtu.tree_map(lambda x: x[0], y_diff_history))
    op = lx.FunctionLinearOperator(
        operator_func,
        input_shape,
        tags=lx.positive_semidefinite_tag,
    )
    return op


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


_Hessian = TypeVar(
    "_Hessian", FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv,
)


class AbstractQuasiNewtonUpdate(eqx.Module):
    """Abstract base class for quasi-Newton updates to the Hessian approximation. An
    update consumes information about the current and preceding step, the gradient
    values at the respective points, and the previous Hessian approximation and returns
    an updated instance of [`optimistix.FunctionInfo.EvalGradHessian`][] or
    [`optimistix.FunctionInfo.EvalGradHessianInv`][], depending on whether we're
    approximating the Hessian or its inverse.
    """

    @abc.abstractmethod
    def __call__(
        self,
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo.EvalGradHessian | FunctionInfo.EvalGradHessianInv,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: HessianUpdateState,
    ) -> tuple[
        FunctionInfo.EvalGradHessian | FunctionInfo.EvalGradHessianInv,
        HessianUpdateState
    ]:
        """Called whenever we want to update the Hessian approximation. This is usually
        in the `accepted` branch of the `step` method of an
        [`optimistix.AbstractQuasiNewton`][] minimiser.

        **Arguments:**

        - `y`: the previous accepted iterate
        - `y_eval`: the current (just accepted) iterate
        - `f_info`: The function value, gradient and Hessian approximation at the
        previous accepted iterate `y`
        - `f_eval_info`: The function value and its gradient at the current (just
            accepted) iterate `y_eval`

        **Returns:**

        The updated Hessian (or Hessian-inverse) approximation.
        """

    @abc.abstractmethod
    def init(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, HessianUpdateState]:
        pass


class _AbstractBFGSDFPUpdate(AbstractQuasiNewtonUpdate):
    """Private intermediate class for BFGS/DFP updates."""

    use_inverse : bool

    def init(self, y: Y, f: Scalar, grad: Y) -> tuple[_Hessian, HessianUpdateState]:
        identity_op = _identity_pytree(y)
        if self.use_inverse:
            f_info = FunctionInfo.EvalGradHessianInv(f, grad, identity_op)
        else:
            f_info = FunctionInfo.EvalGradHessian(f, grad, identity_op)
        return f_info, None

    def no_update(self, inner, grad_diff, y_diff, f_info):
        if self.use_inverse:
            return f_info.hessian_inv
        else:
            return f_info.hessian

    @abc.abstractmethod
    def update(
        self,
        inner: PyTree,
        grad_diff: PyTree,
        y_diff: PyTree,
        f_info: FunctionInfo.EvalGradHessian | FunctionInfo.EvalGradHessianInv,
    ) -> lx.PyTreeLinearOperator:
        ...

    def __call__(
        self,
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo.EvalGradHessian | FunctionInfo.EvalGradHessianInv,
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: HessianUpdateState,
    ) -> tuple[
        FunctionInfo.EvalGradHessian | FunctionInfo.EvalGradHessianInv,
        HessianUpdateState
    ]:
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

        hessian = filter_cond(
            inner_nonzero,
            self.update,
            self.no_update,
            inner,
            grad_diff,
            y_diff,
            f_info,
        )
        if self.use_inverse:
            # in this case `hessian` is the new inverse hessian
            return FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian), None
        else:
            return FunctionInfo.EvalGradHessian(f_eval, grad, hessian), None


class DFPUpdate(_AbstractBFGSDFPUpdate):
    """DFP (Davidon–Fletcher–Powell) approximate Hessian updates.

    This is a variant of the BFGS update.

    See [https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula](https://en.wikipedia.org/wiki/Davidon–Fletcher–Powell_formula).
    """

    use_inverse: bool

    def update(self, inner, grad_diff, y_diff, f_info):
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


DFPUpdate.__init__.__doc__ = """**Arguments:**

- `use_inverse`: The DFP algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves).
"""


class BFGSUpdate(_AbstractBFGSDFPUpdate):
    """BFGS (Broyden–Fletcher–Goldfarb–Shanno) approximate Hessian updates.

    See [https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm).
    """

    use_inverse: bool

    def update(self, inner, grad_diff, y_diff, f_info):
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


BFGSUpdate.__init__.__doc__ = """**Arguments:**

- `use_inverse`: The BFGS algorithm involves computing matrix-vector products of the
    form `B^{-1} g`, where `B` is an approximation to the Hessian of the function to be
    minimised. This means we can either (a) store the approximate Hessian `B`, and do a
    linear solve on every step, or (b) store the approximate Hessian inverse `B^{-1}`,
    and do a matrix-vector product on every step. Option (a) is generally cheaper for
    sparse Hessians (as the inverse may be dense). Option (b) is generally cheaper for
    dense Hessians (as matrix-vector products are cheaper than linear solves).
"""


class _QuasiNewtonState(
    eqx.Module, Generic[Y, Aux, SearchState, DescentState, _Hessian]
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
    hess_update_state: HessianUpdateState



class AbstractQuasiNewton(
    AbstractMinimiser[Y, Aux, _QuasiNewtonState], Generic[Y, Aux, _Hessian]
):
    """Abstract quasi-Newton minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information. These updates to the Hessian are handled by the `hessian_update`
    attribute, which should be a subclass of `AbstractQuasiNewtonUpdate`, such as
    `BFGSUpdate` or `DFPUpdate`.

    This abstract version may be subclassed to choose alternative descent and searches.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: AbstractVar[float]
    atol: AbstractVar[float]
    norm: AbstractVar[Callable[[PyTree], Scalar]]
    descent: AbstractVar[AbstractDescent[Y, _Hessian, Any]]
    search: AbstractVar[AbstractSearch[Y, _Hessian, FunctionInfo.Eval, Any]]
    hessian_update: AbstractVar[AbstractQuasiNewtonUpdate]
    verbose: AbstractVar[frozenset[str]]

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
        f_info, hessian_update_state = self.hessian_update.init(y, f, grad)
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
            hess_update_state=hessian_update_state
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

            f_eval_info, hess_update_state = self.hessian_update(
                y,
                state.y_eval,
                state.f_info,
                FunctionInfo.EvalGrad(f_eval, grad),
                state.hess_update_state
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
                hess_update_state
            )

        def rejected(descent_state):
            return (
                y,
                state.f_info,
                state.aux,
                descent_state,
                jnp.array(False),
                state.hess_update_state
            )

        y, f_info, aux, descent_state, terminate, hess_update_state = filter_cond(
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
            hess_update_state=hess_update_state,
        )
        return y, state, aux

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


class BFGS(AbstractQuasiNewton[Y, Aux, _Hessian]):
    """BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NewtonDescent
    search: BacktrackingArmijo
    hessian_update: AbstractQuasiNewtonUpdate
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
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
        self.search = BacktrackingArmijo()
        self.hessian_update = BFGSUpdate(use_inverse=use_inverse)
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
    line search methods like [`optimistix.ClassicalTrustRegion`][], which use the
    Hessian approximation `B` as part of their own computations.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example
    `verbose=frozenset({"step_size", "loss"})`.
"""


class DFP(AbstractQuasiNewton[Y, Aux, _Hessian]):
    """DFP (Davidon–Fletcher–Powell) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm, whose defining feature is the way
    it progressively builds up a Hessian approximation using multiple steps of gradient
    information.

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
    descent: NewtonDescent
    search: BacktrackingArmijo
    hessian_update: AbstractQuasiNewtonUpdate
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
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
        self.search = BacktrackingArmijo()
        self.hessian_update = DFPUpdate(use_inverse=use_inverse)
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
    line search methods like [`optimistix.ClassicalTrustRegion`][], which use the
    Hessian approximation `B` as part of their own computations.
- `verbose`: Whether to print out extra information about how the solve is
    proceeding. Should be a frozenset of strings, specifying what information to print.
    Valid entries are `step_size`, `loss`, `y`. For example
    `verbose=frozenset({"step_size", "loss"})`.
"""


class _LBFGSState(eqx.Module, strict=True):
    index_start: Array
    y_diff_history: PyTree[Array]
    grad_diff_history: PyTree[Array]
    inner_history: PyTree[Array]


class LBFGSUpdate(AbstractQuasiNewtonUpdate, strict=True):
    """Private intermediate class for LBFGS updates."""

    use_inverse: bool
    history_length: int = 10

    def init(self, y: Y, f: Scalar, grad: Y):
        state = _LBFGSState(
            index_start=jnp.array(0, dtype=int),
            y_diff_history=jtu.tree_map(
                lambda y: jnp.zeros((self.history_length, *y.shape)), y
            ),
            grad_diff_history=jtu.tree_map(
                lambda y: jnp.zeros((self.history_length, *y.shape)), y
            ),
            inner_history=jnp.zeros(10),
        )
        op = _make_lbfgs_operator(
            state.y_diff_history,
            state.grad_diff_history,
            state.inner_history,
            state.index_start
        )
        return FunctionInfo.EvalGradHessianInv(f, grad, op), state

    def no_update(self, inner, grad_diff, y_diff, f_info, start_index):
        return eqx.filter(f_info.hessian_inv, eqx.is_array), jnp.array(0)

    def update(self, inner, grad_diff, y_diff, f_info, start_index):
        assert isinstance(f_info, FunctionInfo.EvalGradHessianInv)
        # update the start index
        dynamic = eqx.filter(_make_lbfgs_operator(
                y_diff,
                grad_diff,
                inner,
                start_index,
            ),
            eqx.is_array
        )
        return dynamic, jnp.array(1)

    def __call__(
        self,
        y: Y,
        y_eval: Y,
        f_info: Union[FunctionInfo.EvalGradHessian, FunctionInfo.EvalGradHessianInv],
        f_eval_info: FunctionInfo.EvalGrad,
        hessian_update_state: HessianUpdateState,
    ) -> tuple[FunctionInfo.EvalGradHessianInv, HessianUpdateState]:
        f_eval = f_eval_info.f
        grad = f_eval_info.grad
        y_diff = (y_eval**ω - y**ω).ω
        grad_diff = (grad**ω - f_info.grad**ω).ω

        y_diff_history = hessian_update_state.y_diff_history
        grad_diff_history = hessian_update_state.grad_diff_history
        index_start = hessian_update_state.index_start
        inner_history = hessian_update_state.inner_history

        # update states
        y_diff_history = jtu.tree_map(
            lambda x, z: x.at[index_start].set(z), y_diff_history, y_diff
        )
        grad_diff_history = jtu.tree_map(
            lambda x, z: x.at[index_start].set(z),
            grad_diff_history, grad_diff
        )
        inner_history = inner_history.at[index_start].set(
            1. / tree_dot(y_diff, grad_diff)
        )
        inner_history = jnp.where(jnp.isinf(inner_history), 0, inner_history)

        # fix the static arguments including the JAXPR.
        # Note that the JAXPR inner var id is overwritten
        # the closure variables are updated correctly, however.
        static = eqx.filter(f_info.hessian_inv, eqx.is_array, inverse=True)

        dynamic, update = filter_cond(
            inner_history[index_start] != 0,
            self.update,
            self.no_update,
            inner_history,
            grad_diff_history,
            y_diff_history,
            f_info,
            index_start,
        )
        hessian = eqx.combine(static, dynamic)

        # increment circular index
        index_start = (index_start + update) % self.history_length

        hessian_update_state = _LBFGSState(
            index_start=jnp.array(index_start),
            y_diff_history=y_diff_history,
            grad_diff_history=grad_diff_history,
            inner_history=inner_history,
        )

        return (
            FunctionInfo.EvalGradHessianInv(f_eval, grad, hessian),
            hessian_update_state
        )

class LBFGS(AbstractQuasiNewton[Y, Aux, _Hessian], strict=True):
    """L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) minimisation algorithm.

    This is a quasi-Newton optimisation algorithm that approximates the inverse Hessian
    using a limited history of gradient and parameter updates. Unlike full BFGS,
    which stores a dense matrix, L-BFGS maintains a memory-efficient
    representation suitable for large-scale problems.

    Supports the following `options`:

    - `autodiff_mode`: whether to use forward- or reverse-mode autodifferentiation to
        compute the gradient. Can be either `"fwd"` or `"bwd"`. Defaults to `"bwd"`,
        which is usually more efficient. Changing this can be useful when the target
        function does not support reverse-mode automatic differentiation.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: NewtonDescent
    search: AbstractSearch
    hessian_update: AbstractQuasiNewtonUpdate
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
        search: AbstractSearch = BacktrackingArmijo(),
        history_length: int = 10,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        # TODO(raderj): switch out `BacktrackingArmijo` with a better line search.
        self.search = search
        self.hessian_update = LBFGSUpdate(
            use_inverse=True,
            history_length=history_length
        )
        self.verbose = verbose
