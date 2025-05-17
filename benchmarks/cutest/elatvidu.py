import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class ELATVIDU(AbstractUnconstrainedMinimisation, strict=True):
    """The ELATVIDU (El-Attar-Vidyasagar-Dutta) function.

    SCIPY global optimization benchmark example ElAttarVidyasagarDutta
    Fit: (x_1^2 + x_2 - 10, x_1 + x_2^2 - 7, x_1^2 + x_2^3 - 1) + e = 0

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould

    Classification: SUR2-MN-2-0
    """

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Computing the residuals according to the problem definition
        r1 = x1**2 + x2 - 10
        r2 = x1 + x2**2 - 7
        r3 = x1**2 + x2**3 - 1

        # Sum of squared residuals
        return r1**2 + r2**2 + r3**2

    def y0(self):
        # Initial values from SIF file
        return jnp.array([1.0, 5.0])

    def args(self):
        return None

    def expected_result(self):
        return None  # Solution not provided in the SIF file

    def expected_objective_value(self):
        return jnp.array(0.0)  # From OBJECT BOUND in the file
