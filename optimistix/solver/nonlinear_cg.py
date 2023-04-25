from typing import ClassVar

import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from ..line_search import AbstractProxy


# WARNING: Implementation still WIP.
class NonlinearCG(AbstractProxy):
    needs_gradient: ClassVar[bool] = True
    needs_hessian: ClassVar[bool] = False

    def __call__(
        self, delta, delta_args, problem, y, args, state, options, vector, operator
    ):
        #
        # WARNING: perhaps there is a nice way around this, but I am going
        # to assume that we have, in the state, access to the prior gradient
        # and prior step. Can we store this information in the class itself?
        #
        try:
            method = options["method"]
        except KeyError:
            # NOTE: should we keep a default method here or throw an error?
            method = "fletcher-reeves"

        if not isinstance(method, str):
            raise ValueError(
                f"The method {method} passed to NonlinearCG is not \
                a string. NonlinearCG expected one of 'fletcher-reeves', \
                'polak-ribiere', 'hestenes-stiefel', or 'dai-yuan'."
            )

        method = method.lower()
        implemented = [
            "fletcher-reeves",
            "polak-ribiere",
            "hestenes-stiefel",
            "dai-yan",
        ]
        if not any(method in _method for _method in implemented):
            raise ValueError(
                f"The method {method} passed to NonlinearCG is not \
                a valid method. NonlinearCG expected one of 'fletcher-reeves', \
                'polak-ribiere', 'hestenes-stiefel', or 'dai-yuan'."
            )

        _norm_squared = lambda tree: jtu.tree_reduce(
            lambda x, y: x + y, ω(tree).call(lambda x: jnp.sum(x**2)).ω
        )
        _sum_leaves = lambda tree: jtu.tree_reduce(lambda x, y: x + y, tree)

        #
        # aside from all the PyTree handling, this is just the implementation
        # of these methods described on the wikipedia page for nonlinear
        # cg: https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
        # note that state.vector differs by a factor of `-1` from the wiki grad
        # so we do not have the `-` in the denominator of Dai-Yuan or
        # Hestenes-Stiefel.
        #
        if method in "fletcher-reeves":
            beta = _norm_squared(vector) / _norm_squared(state.prev_vector)

        if method in "polak-ribiere":
            grad_diff = (ω(vector) - ω(state.prev_vector)) * ω(state.vector)
            grad_diff = grad_diff.call(jnp.sum).ω
            beta = _sum_leaves(grad_diff) / _norm_squared(state.prev_vector)

        if method in "hestenes-stiefel":
            diff = ω(vector) - ω(state.prev_vector)

            grad_diff = (ω(vector) * diff).call(jnp.sum).ω
            dir_diff = (ω(state.prev_dir) * diff).call(jnp.sum).ω

            beta = _sum_leaves(grad_diff) / _sum_leaves(dir_diff)

        if method in "dai-yuan":
            dir_diff = ω(state.prev_dir) * (ω(vector) - ω(state.prev_vector))
            dir_diff = dir_diff.call(jnp.sum).ω

            beta = _norm_squared(vector) / _sum_leaves(dir_diff)

        return -vector + beta * ω(state.prev_dir)
