import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This problem still has to be verified against another CUTEst interface.
class ARWHEAD(AbstractUnconstrainedMinimisation, strict=True):
    """The ARWHEAD function.

    A quartic problem whose Hessian is an arrow-head (downwards) with diagonal central
    part and border-width of 1.

    Source: Problem 55 in
    A.R. Conn, N.I.M. Gould, M. Lescrenier and Ph.L. Toint,
    "Performance of a multifrontal scheme for partially separable optimization",
    Report 88/4, Dept of Mathematics, FUNDP (Namur, B), 1988.

    SIF input: Ph. Toint, Dec 1989.
    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # SIF file lists 100, 500, 1000, 5000 as suggested dimensions

    def objective(self, y, args):
        del args
        yn = y[-1]
        f1 = -4 * y[:-1] ** 2 + 3
        f2 = (y[:-1] ** 2 + yn**2) ** 2
        return jnp.sum(f1 + f2)

    def y0(self):
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.0)
