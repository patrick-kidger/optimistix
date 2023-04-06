import jax

import optimistix as optx

from ..custom_types import sentinel


def init_derivatives(problem, y, needs_gradient, needs_hessian, options):
    if needs_hessian or needs_gradient:
        jrev = jax.jacrev(problem.fn, has_aux=problem.has_aux)
    if needs_hessian and not needs_gradient:
        try:
            hessian = options["hessian"]
            gradient = sentinel
        except KeyError:
            gradient = sentinel
            hessian = jax.jacfwd(jrev, has_aux=problem.has_aux)(y)
            hessian = optx.MatrixLinearOperator(
                hessian, tags=(optx.positive_semidefinite_tag, optx.symmetric_tag)
            )
    elif needs_hessian and needs_gradient:
        # WARNING: If you want to pass a gradient but calculate the
        # hessian in here, tough luck.
        try:
            hessian = options["hessian"]
            gradient = options["gradient"]
        except KeyError:
            gradient = jrev(y)
            hessian = jax.jacfwd(jrev)(y)
    elif needs_gradient:
        try:
            gradient = options["gradient"]
            hessian = sentinel
        except KeyError:
            gradient = jrev(y)
            hessian = sentinel
    else:
        gradient = sentinel
        hessian = sentinel

    return gradient, hessian
