import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class GROWTHLS(AbstractUnconstrainedMinimisation):
    """GROWTHLS - Growth problem in 3 variables.

    This problem involves fitting the observed growth g(n) from Gaussian
    Elimination with complete pivoting to a function of the form:

    U1 * n ** (U2 + LOG(n) * U3)

    Source: NIST nonlinear regression test set

    SIF input: Nick Gould, Nov, 1991, modified by Ph. Toint, March 1994.

    Classification: SUR2-AN-3-0
    """

    def objective(self, y, args):
        del args

        # Extract the 3 parameters
        u1, u2, u3 = y

        # Data points: n values (input)
        n_values = jnp.array(
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 25.0]
        )

        # Observed growth values g(n) (output)
        g_values = jnp.array(
            [
                8.4305,
                9.5294,
                10.4627,
                12.0,
                13.0205,
                14.5949,
                16.1078,
                18.0596,
                20.4569,
                24.25,
                32.9863,
            ]
        )

        # Add the missing value for n=8 (based on the constant section)
        g_values = jnp.array(
            [
                8.0,
                8.4305,
                9.5294,
                10.4627,
                12.0,
                13.0205,
                14.5949,
                16.1078,
                18.0596,
                20.4569,
                24.25,
                32.9863,
            ]
        )

        # Define the model function: u1 * n ** (u2 + log(n) * u3)
        def model_func(n):
            log_n = jnp.log(n)
            power = u2 + log_n * u3
            return u1 * n**power

        # Compute model predictions for all n values
        predictions = jax.vmap(model_func)(n_values)

        # Compute residuals
        residuals = predictions - g_values

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Starting point from the SIF file: U1 = 100.0, U2 and U3 not specified
        # Using reasonable values for U2 and U3
        return jnp.array([100.0, 0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not specified in the SIF file
        return None

    def expected_objective_value(self):
        # The minimum objective value is not specified in the SIF file
        return None
