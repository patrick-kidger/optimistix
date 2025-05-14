import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: this has not yet been compared against another interface to CUTEst
class FLETCBV3(AbstractUnconstrainedMinimisation, strict=True):
    """The FLETCBV3 function.

    A boundary value problem from Fletcher (1992).

    Source: The first problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    Note J. Haffner --------------------------------------------------------------------
    The reference given appears to be incorrect, the PDF available under the title above
    does not include a problem description.

    This can be defined for different dimensions (original SIF allows 10, 100, 1000,
    5000, or 10000), with 5000 being the default in the SIF file.
    ------------------------------------------------------------------------------------

    SIF input: Nick Gould, Oct 1992.
    Classification: OUR2-AN-V-0
    """

    n: int = 5000
    scale: float = 1e8  # Called OBJSCALE in the SIF file
    extra_term: int = 1  # Corresponds to the parameter kappa, which is 1 or 0

    def objective(self, y, args):
        p, c = args
        h = 1.0 / (self.n + 1)
        h2 = h * h

        f1 = (y[0] + y[1]) ** 2
        f2 = jnp.sum((y[:-1] - y[1:]) ** 2)
        f3 = (h2 + 2) / h2 * jnp.sum(y)
        f4 = c / h2 * jnp.sum(jnp.cos(y))

        return 0.5 * p * (f1 + f2) - p * f3 + f4

    def y0(self):
        n = self.n
        h = 1.0 / (self.n + 1)
        # Starting point according to SIF file: i*h for i=1..n
        return jnp.arange(1, n + 1) * h

    def args(self):
        # p and kappa from SIF file
        p = 1 / self.scale
        c = self.extra_term
        return jnp.array([p, c])

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None  # Takes different values for different problem configurations
