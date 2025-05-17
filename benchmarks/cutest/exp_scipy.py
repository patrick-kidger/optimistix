import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class EXP2(AbstractUnconstrainedMinimisation, strict=True):
    """The EXP2 function.

    SCIPY global optimization benchmark example Exp2
    Fit: y = e^{-i/10 x_1} - 5e^{-i/10 x_2} - e^{-i/10} + 5e^{-i} + e

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould

    Classification: SUR2-MN-2-0
    """

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Define a function to compute a single residual given an index
        def compute_residual(i):
            term1 = jnp.exp(-i / 10 * x1)
            term2 = -5 * jnp.exp(-i / 10 * x2)
            term3 = -jnp.exp(-i / 10)
            term4 = 5 * jnp.exp(-i)
            term5 = jnp.exp(1)  # e constant
            return term1 + term2 + term3 + term4 + term5

        # Vectorize the residual computation across all indices
        indices = jnp.arange(10.0)
        residuals = jax.vmap(compute_residual)(indices)

        # Sum of squared residuals
        return jnp.sum(jnp.square(residuals))

    def y0(self):
        # Initial values from SIF file
        return jnp.array([1.0, 5.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class EXP2B(AbstractUnconstrainedMinimisation, strict=True):
    """The EXP2B function.

    SCIPY global optimization benchmark example Exp2
    Fit: y = e^{-i/10 x_1} - 5e^{-i/10 x_2} - e^{-i/10} + 5e^{-i} + e
    Version with box-constrained feasible region: 0 <= x <= 20

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, July 2021

    Classification: SBR2-MN-2-0
    """

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args

        # Handle box constraints: 0 <= x <= 20
        x1, x2 = jnp.clip(y, 0.0, 20.0)

        # Define a function to compute a single residual given an index
        def compute_residual(i):
            term1 = jnp.exp(-i / 10 * x1)
            term2 = -5 * jnp.exp(-i / 10 * x2)
            term3 = -jnp.exp(-i / 10)
            term4 = 5 * jnp.exp(-i)
            term5 = jnp.exp(1)  # e constant
            return term1 + term2 + term3 + term4 + term5

        # Vectorize the residual computation across all indices
        indices = jnp.arange(10.0)
        residuals = jax.vmap(compute_residual)(indices)

        # Sum of squared residuals
        return jnp.sum(jnp.square(residuals))

    def y0(self):
        # Initial values from SIF file
        return jnp.array([1.0, 5.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class EXP2NE(AbstractUnconstrainedMinimisation, strict=True):
    """The EXP2NE function.

    SCIPY global optimization benchmark example Exp2
    Fit: y = e^{-i/10 x_1} - 5e^{-i/10 x_2} - e^{-i/10} + 5e^{-i} + e
    Nonlinear-equation formulation of EXP2

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould

    Classification: NOR2-MN-2-10
    """

    n: int = 2  # Problem has 2 variables

    def objective(self, y, args):
        del args
        x1, x2 = y

        # Define a function to compute a single residual given an index
        def compute_residual(i):
            term1 = jnp.exp(-i / 10 * x1)
            term2 = -5 * jnp.exp(-i / 10 * x2)
            term3 = -jnp.exp(-i / 10)
            term4 = 5 * jnp.exp(-i)
            term5 = jnp.exp(1)  # e constant
            return term1 + term2 + term3 + term4 + term5

        # Vectorize the residual computation across all indices
        indices = jnp.arange(10.0)
        residuals = jax.vmap(compute_residual)(indices)

        # Sum of squared residuals
        return jnp.sum(jnp.square(residuals))

    def y0(self):
        # Initial values from SIF file
        return jnp.array([1.0, 5.0])

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
