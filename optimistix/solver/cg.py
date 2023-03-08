from typing import Callable, Optional

import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from ..custom_types import Scalar
from ..linear_operator import (
    AbstractLinearOperator,
    IdentityLinearOperator,
    is_negative_semidefinite,
    is_positive_semidefinite,
    linearise,
)
from ..linear_solve import AbstractLinearSolver
from ..misc import max_norm
from ..solution import RESULTS


def _tree_dot(a: PyTree[Array], b: PyTree[Array]) -> Scalar:
    a = jtu.tree_leaves(a)
    b = jtu.tree_leaves(b)
    assert len(a) == len(b)
    return sum(
        [jnp.vdot(ai, bi, precision=lax.Precision.HIGHEST) for ai, bi in zip(a, b)]
    )


# TODO(kidger): this is pretty slow to compile.
# - CG evaluates `operator.mv` three times.
# - Normal CG evaluates `operator.mv` seven (!) times.
# Possibly this can be cheapened a bit somehow?
class CG(AbstractLinearSolver):
    rtol: float
    atol: float
    normal: bool = False
    norm: Callable[[PyTree], Scalar] = max_norm
    stabilise_every: Optional[int] = 10
    max_steps: Optional[int] = None

    def __post_init__(self):
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if self.atol == 0 and self.rtol == 0 and self.max_steps is None:
            raise ValueError(
                "Must specify `rtol`, `atol`, or `max_steps` (or some combination of "
                "all three)."
            )

    def init(self, operator, options):
        del options
        is_nsd = is_negative_semidefinite(operator)
        if not self.normal:
            if operator.in_structure() != operator.out_structure():
                raise ValueError(
                    "`CG(..., normal=False)` may only be used for linear solves with "
                    "square matrices."
                )
            if not (is_positive_semidefinite(operator) | is_nsd):
                raise ValueError(
                    "`CG(..., normal=False)` may only be used for positive "
                    "or negative definite linear operators"
                )
            if is_nsd:
                operator = -operator
        return operator, is_nsd

    # This differs from jax.scipy.sparse.linalg.cg in:
    # 1. Every few steps we calculate the residual directly, rather than by cheaply
    #    using the existing quantities. This improves numerical stability.
    # 2. We use a more sophisticated termination condition. To begin with we have an
    #    rtol and atol in the conventional way, inducing a vector-valued scale. This is
    #    then checked in both the `y` and `b` domains (for `Ay = b`).
    # 3. We return the number of steps, and whether or not the solve succeeded, as
    #    additional information.
    # 4. We don't try to support complex numbers. (Yet.)
    def compute(self, state, vector, options):
        operator, is_nsd = state
        if self.normal:
            # Linearise if JacobianLinearOperator, to avoid computing the forward
            # pass separately for mv and transpose_mv.
            # This choice is "fast by default", even at the expense of memory.
            # If a downstream userc wants to avoid this then they can call
            # `linear_solve(operator.T @ operator, operator.mv(b), solver=CG(..., normal=False)`  # noqa: E501
            # directly.
            operator = linearise(operator)

            _mv = operator.mv
            _transpose_mv = operator.transpose().mv

            def mv(v):
                return _transpose_mv(_mv(v))

            vector = _transpose_mv(vector)
        else:
            mv = operator.mv
        structure = operator.in_structure()
        del operator

        try:
            preconditioner = options["preconditioner"]
        except KeyError:
            preconditioner = IdentityLinearOperator(structure)
        else:
            if not isinstance(preconditioner, AbstractLinearOperator):
                raise ValueError("The preconditioner must be a linear operator.")
            if preconditioner.in_structure() != structure:
                raise ValueError(
                    "The preconditioner must have `in_structure` that matches the "
                    "operator's `in_strucure`."
                )
            if preconditioner.out_structure() != structure:
                raise ValueError(
                    "The preconditioner must have `out_structure` that matches the "
                    "operator's `in_structure`."
                )
            if not is_positive_semidefinite(preconditioner):
                raise ValueError("The preconditioner must be positive definite.")
        try:
            y0 = options["y0"]
        except KeyError:
            y0 = jtu.tree_map(jnp.zeros_like, vector)
        else:
            if jax.eval_shape(lambda: y0)() != jax.eval_shape(lambda: vector)():
                raise ValueError(
                    "`y0` must have the same structure, shape, and dtype as `vector`"
                )

        r0 = (vector**ω - mv(y0) ** ω).ω
        p0 = preconditioner.mv(r0)
        gamma0 = _tree_dot(r0, p0)
        initial_value = (jnp.zeros_like(y0), y0, r0, p0, gamma0, 0)
        has_scale = not (
            isinstance(self.atol, (int, float))
            and isinstance(self.rtol, (int, float))
            and self.atol == 0
            and self.rtol == 0
        )
        if has_scale:
            b_scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω

        def cond_fun(value):
            diff, y, r, _, gamma, step = value
            out = gamma > 0
            if self.max_steps is not None:
                out = out & (step < self.max_steps)
            if has_scale:
                # i.e. given Ay=b, then we have to be doing better than `scale` in both
                # the `y` and the `b` spaces.
                y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
                norm1 = self.norm((r**ω / b_scale**ω).ω)
                norm2 = self.norm((diff**ω / y_scale**ω).ω)
                out = out & ((norm1 > 1) | (norm2 > 1))
            return out

        def body_fun(value):
            _, y, r, p, gamma, step = value
            mat_p = mv(p)
            alpha = gamma / _tree_dot(p, mat_p)
            diff = (alpha * p**ω).ω
            y = (y**ω + diff**ω).ω
            step = eqxi.nonbatchable(step)
            step = step + 1

            # E.g. see B.2 of
            # https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            # We compute the residual the "expensive" way every now and again, so as to
            # correct numerical rounding errors.
            def stable_r():
                return (vector**ω - mv(y) ** ω).ω

            def cheap_r():
                return (r**ω - alpha * mat_p**ω).ω

            if self.stabilise_every == 1:
                r = stable_r()
            elif self.stabilise_every is None:
                r = cheap_r()
            else:
                stable_step = (step % self.stabilise_every) == 0
                r = lax.cond(stable_step, stable_r, cheap_r)

            z = preconditioner.mv(r)
            gamma_prev = gamma
            gamma = _tree_dot(r, z)
            beta = gamma / gamma_prev
            p = (z**ω + beta * p**ω).ω
            return diff, y, r, p, gamma, step

        _, solution, _, _, _, num_steps = lax.while_loop(
            cond_fun, body_fun, initial_value
        )
        if self.max_steps is None:
            result = RESULTS.successful
        else:
            result = jnp.where(
                (num_steps == self.max_steps),
                RESULTS.max_steps_reached,
                RESULTS.successful,
            )

        if is_nsd and not self.normal:
            solution = -(solution**ω).ω
        return (
            solution,
            result,
            {"num_steps": num_steps, "max_steps": self.max_steps},
        )

    def pseudoinverse(self, operator):
        return True

    def transpose(self, state, options):
        # In particular the preconditioner is necessarily already positive definite, so
        # it doesn't need transposing or anything.
        return state, options
