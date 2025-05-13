import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Needs verification against another CUTEst interface
class COSINE(AbstractUnconstrainedMinimisation, strict=True):
    """The COSINE function.

    A function with nontrivial groups and repetitious elements.

    Source: N. Gould, private communication.

    SIF input: N. Gould, Jan 1996

    Classification: OUR2-AN-V-0
    """

    n: int = 1000  # Other suggested dimensions 10, 100, 10000

    def objective(self, y, args):
        del args
        return jnp.sum(jnp.cos(-0.5 * y[1:] + y[:-1] ** 2))

    def y0(self):
        # Initial guess - specified as 1.0 for all variables
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
