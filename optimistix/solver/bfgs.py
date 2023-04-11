import jax
import jax.numpy as jnp
from equinox.internal import ω

from ..custom_types import sentinel
from ..line_search import AbstractGLS, AbstractModel
from ..linear_operator import PyTreeLinearOperator
from ..minimise import minimise
from .models import UnnormalizedNewton
from .quasi_newton import AbstractQuasiNewton, QNState


#
# Right now, this is performing the update `B_k -> B_(k + 1)` which solves
# `B_k p = g`. We could just as well update `Binv_k -> Binv_(k + 1)`, ie.
# `p = Binv_k g` is just the matrix vector product.
#
# To change to this version is straightforward, change update_state below
# to make the correct BFGS update, and then create a model which returns as
# its descent direction the mvp
# ```
# model.descent_dir(state, ...) = state.operator.mv(g)
# ```
#


class BFGS(AbstractQuasiNewton):
    line_search: AbstractGLS = sentinel
    model: AbstractModel = sentinel

    def __post_init__(self):
        if self.line_search == sentinel:
            raise ValueError("No line search initialized in BFGS")

        if self.model == sentinel:
            self.model = UnnormalizedNewton(gauss_newton=False)

    def step(self, problem, y, args, options, state):
        if self.model.gauss_newton:
            raise ValueError(
                "A model with gauss_newton=True was passed to GeneralBFGS \
                which is not a Gauss-Newton method."
            )
        line_search = self.line_search(self.model)

        if state.vector != sentinel and state.vector != sentinel:
            options["gradient"] = state.vector
            options["hessian"] = state.operator

        sol = minimise(problem, line_search, y, args, options)
        new_y, new_state = self.update_solution(problem, y, sol, state)

        return new_y, new_state, sol.state.aux

    def update_state(self, problem, y, sol, state):
        # TODO(raderj): find a way to remove this call to problem.fn.
        # it should be possible by calling step with no gradient and
        # using the resulting vector. I just want to avoid the whole line
        # search...
        new_grad = jax.jacrev(problem.fn, has_aux=problem.has_aux)(y)

        diff = sol.state.decrease_dir
        t_diff = new_grad - sol.state.vector

        if state.operator == sentinel:
            new_hess = sol.state.operator
        else:
            (t_diff_ravel, _) = jax.flatten_util.ravel_pytree(t_diff)
            (diff_ravel, _) = jax.flatten_util.ravel_pytree(diff)
            hess_mv = state.operator.mv(diff)
            term1 = (t_diff_ravel @ t_diff_ravel.T) / (t_diff_ravel.T @ diff_ravel)
            term2 = (hess_mv @ hess_mv.T) / (diff_ravel.T @ hess_mv)
            new_hess = PyTreeLinearOperator(
                state.operator.as_matrix() + term1 - term2,
                jax.eval_shape(lambda: new_grad),
            )

        new_y = (ω(y) + diff).ω
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        diffsize = self.norm((ω(sol.state.descent_dir) / ω(scale)).ω)
        new_state = QNState(
            step=state.step + 1,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=sol.result,
            vector=new_hess,
            operator=sentinel,
        )
        return new_y, new_state
