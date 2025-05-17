import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DEVGLA1(AbstractUnconstrainedMinimisation):
    """DeVilliers-Glasser problem 1 from the SCIPY global optimization benchmark.

    This problem involves fitting a model of the form:
    y = x₁ * x₂^t * sin(t * x₃ + x₄) + e

    to a set of data points (t, y) where t ranges from 0.1 to 2.4 in steps of 0.1
    and the y values are pre-computed based on specific parameter values.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    Classification: SUR2-MN-4-0
    """

    n: int = 4  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3, x4 = y

        # Generate time points t from 0.1 to 2.4 with step 0.1
        t_values = jnp.arange(1, 25) * 0.1

        # Compute the true data points
        true_values = jnp.array(
            [
                60.137,
                61.0928,
                62.4274,
                64.2380,
                66.6387,
                69.6393,
                73.3182,
                77.7407,
                82.9391,
                89.0088,
                95.9844,
                104.011,
                113.212,
                123.722,
                135.682,
                149.243,
                164.568,
                181.838,
                201.251,
                223.028,
                247.403,
                274.637,
                304.997,
                338.774,
            ]
        )

        # Compute the model predictions
        model_pred = self._model(x1, x2, x3, x4, t_values)

        # Calculate residuals
        residuals = model_pred - true_values

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def _model(self, x1, x2, x3, x4, t_values):
        """Model function: x₁ * x₂^t * sin(t * x₃ + x₄)"""
        # Compute x₂^t for all t values
        x2_power_t = x2**t_values

        # Compute sin(t * x₃ + x₄) for all t values
        sin_term = jnp.sin(t_values * x3 + x4)

        # Compute the final model prediction
        return x1 * x2_power_t * sin_term

    def y0(self):
        # Initial values from SIF file
        return jnp.array([2.0, 2.0, 2.0, 2.0])

    def args(self):
        return None

    def expected_result(self):
        # True parameter values that generate the data
        # These values aren't explicitly given in the SIF file,
        # but can be derived from the data generation process
        return jnp.array([60.137, 1.371, 3.112, 1.761])

    def expected_objective_value(self):
        # For a perfect fit, the objective value would be 0
        return jnp.array(0.0)


# TODO: human review required
class DEVGLA2(AbstractUnconstrainedMinimisation):
    """DeVilliers-Glasser problem 2 from the SCIPY global optimization benchmark.

    This problem involves fitting a model of the form:
    y = x₁ * x₂^t * tanh(t * x₃ + sin(t * x₄)) * cos(t * e^x₅) + e

    to a set of data points (t, y) where t ranges from 0.1 to 1.6 in steps of 0.1
    and the y values are pre-computed based on specific parameter values.

    Source: Problem from the SCIPY benchmark set
    https://github.com/scipy/scipy/tree/master/benchmarks/benchmarks/go_benchmark_functions

    SIF input: Nick Gould, Jan 2020

    Classification: SUR2-MN-5-0
    """

    n: int = 5  # Number of variables

    def objective(self, y, args):
        del args
        x1, x2, x3, x4, x5 = y

        # Generate time points t from 0.1 to 1.6 with step 0.1
        t_values = jnp.arange(1, 17) * 0.1

        # Compute the true data points
        true_values = jnp.array(
            [
                53.81,
                53.8196,
                53.8184,
                53.7896,
                53.7134,
                53.5765,
                53.3557,
                53.0372,
                52.6090,
                52.0546,
                51.3607,
                50.5049,
                49.4693,
                48.2468,
                46.8394,
                45.2522,
            ]
        )

        # Compute the model predictions
        model_pred = self._model(x1, x2, x3, x4, x5, t_values)

        # Calculate residuals
        residuals = model_pred - true_values

        # Sum of squared residuals
        return jnp.sum(residuals**2)

    def _model(self, x1, x2, x3, x4, x5, t_values):
        """Model function: x₁ * x₂^t * tanh(t * x₃ + sin(t * x₄)) * cos(t * e^x₅)"""
        # Compute x₂^t for all t values
        x2_power_t = x2**t_values

        # Compute sin(t * x₄) for all t values
        sin_tx4 = jnp.sin(t_values * x4)

        # Compute tanh(t * x₃ + sin(t * x₄)) for all t values
        tanh_term = jnp.tanh(t_values * x3 + sin_tx4)

        # Compute cos(t * e^x₅) for all t values
        cos_term = jnp.cos(t_values * jnp.exp(x5))

        # Compute the final model prediction
        return x1 * x2_power_t * tanh_term * cos_term

    def y0(self):
        # Initial values from SIF file
        return jnp.array([20.0, 2.0, 2.0, 2.0, 0.2])

    def args(self):
        return None

    def expected_result(self):
        # Solution values given in the SIF file
        return jnp.array([53.81, 1.27, 3.012, 2.13, 0.507])

    def expected_objective_value(self):
        # For a perfect fit, the objective value would be 0
        return jnp.array(0.0)
