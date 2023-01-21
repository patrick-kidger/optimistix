from typing import Callable, Optional

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from ..custom_types import Scalar
from ..linear_operator import AbstractLinearOperator, IdentityLinearOperator
from ..linear_solve import AbstractLinearSolver
from ..misc import rms_norm
from ..solution import RESULTS


def _tree_dot(a: PyTree[Array], b: PyTree[Array]) -> Scalar:
    a = jax.tree_leaves(a)
    b = jax.tree_leaves(b)
    assert len(a) == len(b)
    return sum([jnp.sum(ai * bi) for ai, bi in zip(a, b)])


# TODO(kidger): this is pretty slow to compile.
# - CG evaluates `operator.mv` twice.
# - Normal CG evaluates `operator.mv` six times.
# Possibly this can be cheapend a bit somehow?
class CG(AbstractLinearSolver):
    rtol: float
    atol: float
    norm: Callable = rms_norm
    materialise: bool = False
    normal: bool = False
    maybe_singular: bool = True
    max_steps: Optional[int] = None

    def is_maybe_singular(self):
        return self.maybe_singular

    def will_materialise(self, operator):
        return self.materialise

    def init(self, operator, options):
        del options
        if not self.normal:
            if operator.in_structure() != operator.out_structure():
                raise ValueError(
                    "`CG(..., normal=False)` may only be used for linear solves with "
                    "square matrices"
                )
            if not operator.pattern.symmetric:
                raise ValueError(
                    "`CG(..., normal=False)` may only be used for symmetric linear "
                    "operators"
                )
        if self.materialise:
            operator = operator.materialise()
        return operator

    # This differs from jax.scipy.sparse.linalg.cg in:
    # 1. We use a slightly different termination condition -- rtol and atol are used in
    #    a conventional way, and `scale` is vector-valued (instead of scalar-valued).
    # 2. We return the number of steps, and whether or not the solve succeeded, as
    #    additional information.
    # 3. We don't try to support complex numbers. (Yet.)
    # 4. We support PyTree-valued state.
    def compute(self, state, vector, options):
        if self.normal:
            _transpose_mv = jax.linear_transpose(state.mv, state.in_structure())
            mv = lambda v, *, _state=state: _transpose_mv(_state.mv(v))
            vector = _transpose_mv(vector)
        else:
            mv = state.mv
        structure = state.in_structure()
        del state

        try:
            preconditioner = options["preconditioner"]
        except KeyError:
            preconditioner = IdentityLinearOperator(structure)
        else:
            if not isinstance(preconditioner, AbstractLinearOperator):
                raise ValueError("preconditioner must be a linear operator")
            if preconditioner.in_structure() != structure:
                raise ValueError(
                    "preconditioner must have `in_structure` that matches the "
                    "operator's `in_strucure`"
                )
            if preconditioner.out_structure() != structure:
                raise ValueError(
                    "preconditioner must have `out_structure` that matches the "
                    "operator's `in_strucure`"
                )
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
        scale = (self.atol + self.rtol * ω(vector).call(jnp.abs)).ω
        initial_value = (y0, r0, p0, gamma0, 0)

        def cond_fun(value):
            _, r, _, _, step = value
            out = self.norm((r**ω / scale**ω).ω) > 1
            if self.max_steps is not None:
                out = out & (step < self.max_steps)
            return out

        def body_fun(value):
            y, r, p, gamma, step = value
            mat_p = mv(p)
            alpha = gamma / _tree_dot(p, mat_p)
            y = (y**ω + alpha * p**ω).ω
            r = (r**ω - alpha * mat_p**ω).ω
            z = preconditioner.mv(r)
            gamma_prev = gamma
            gamma = _tree_dot(r, z)
            beta = gamma / gamma_prev
            p = (z**ω + beta * p**ω).ω
            return y, r, p, gamma, step + 1

        solution, _, _, _, num_steps = lax.while_loop(cond_fun, body_fun, initial_value)
        if self.max_steps is None:
            result = RESULTS.successful
        else:
            result = jnp.where(
                num_steps == self.max_steps,
                RESULTS.max_steps_reached,
                RESULTS.successful,
            )
        return solution, result, {"num_steps": num_steps, "max_steps": self.max_steps}

    def transpose(self, state, options):
        transpose_state = state.transpose()
        try:
            preconditioner = options["preconditioner"]
        except KeyError:
            transpose_options = {}
        else:
            transpose_options = dict(preconditioner=preconditioner.transpose())
        return transpose_state, transpose_options
