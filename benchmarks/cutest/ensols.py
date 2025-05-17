import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class ENSOLS(AbstractUnconstrainedMinimisation):
    """ENSOLS - El Niño-Southern Oscillation data fitting.

    This problem involves fitting a sum of sinusoids with different periods
    to the El Niño-Southern Oscillation (ENSO) time series data, which measures
    the monthly averaged atmospheric pressure differences between Easter Island
    and Darwin, Australia.

    The model function is:
        y = b1 + b2*cos(2πt/12) + b3*sin(2πt/12) +
            b5*cos(2πt/b4) + b6*sin(2πt/b4) +
            b8*cos(2πt/b7) + b9*sin(2πt/b7)

    Source: NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference:
    Kahaner, D., C. Moler, and S. Nash, (1989).
    Numerical Methods and Software.
    Englewood Cliffs, NJ: Prentice Hall, pp. 441-445.

    SIF input: Ph. Toint, Apr 1997.

    Classification: SUR2-MN-9-0
    """

    # Allow selecting which starting point to use (0-based indexing)
    start_point: int = 0  # 0 or 1
    _allowed_start_points = frozenset({0, 1})

    def __check_init__(self):
        if self.start_point not in self._allowed_start_points:
            allowed = self._allowed_start_points
            msg = f"start_point must be in {allowed}, got {self.start_point}"
            raise ValueError(msg)

    def objective(self, y, args):
        del args

        # Extract the 9 parameters
        b1, b2, b3, b4, b5, b6, b7, b8, b9 = y

        # Time points: 1 to 168 months
        t = jnp.arange(1, 169)

        # ENSO observed data from the SIF file
        enso_data = jnp.array(
            [
                12.9,
                11.3,
                10.6,
                11.2,
                10.9,
                7.5,
                7.7,
                11.7,
                12.9,
                14.3,
                10.9,
                13.7,
                17.1,
                14.0,
                15.3,
                8.5,
                5.7,
                5.5,
                7.6,
                8.6,
                7.3,
                7.6,
                12.7,
                11.0,
                12.7,
                12.9,
                13.0,
                10.9,
                10.4,
                10.2,
                8.0,
                10.9,
                13.6,
                10.5,
                9.2,
                12.4,
                12.7,
                13.3,
                10.1,
                7.8,
                4.8,
                3.0,
                2.5,
                6.3,
                9.7,
                11.6,
                8.6,
                12.4,
                10.5,
                13.3,
                10.4,
                8.1,
                3.7,
                1.0,
                1.2,
                2.3,
                4.5,
                6.6,
                5.0,
                10.3,
                9.2,
                12.1,
                10.3,
                9.5,
                5.3,
                3.8,
                2.0,
                0.8,
                0.2,
                0.9,
                2.0,
                6.7,
                7.9,
                11.2,
                8.4,
                7.1,
                5.6,
                3.5,
                1.0,
                1.2,
                1.5,
                5.1,
                3.3,
                9.8,
                9.1,
                11.9,
                10.4,
                9.5,
                4.4,
                2.9,
                -0.3,
                1.4,
                3.5,
                5.3,
                8.2,
                11.0,
                11.8,
                11.8,
                8.3,
                9.8,
                5.3,
                3.9,
                1.9,
                3.7,
                3.5,
                5.3,
                5.9,
                9.1,
                9.4,
                12.2,
                9.5,
                7.1,
                4.8,
                2.9,
                1.2,
                0.1,
                -0.8,
                1.9,
                0.3,
                6.6,
                9.9,
                12.5,
                9.9,
                6.4,
                5.1,
                3.6,
                1.8,
                0.7,
                0.4,
                1.4,
                2.5,
                5.5,
                8.7,
                14.3,
                10.0,
                6.5,
                3.0,
                2.4,
                2.0,
                0.8,
                0.9,
                5.1,
                5.5,
                8.0,
                10.3,
                13.5,
                9.8,
                5.2,
                1.8,
                0.5,
                1.3,
                3.9,
                5.0,
                5.9,
                5.6,
                6.9,
                10.1,
                11.5,
                9.5,
                6.4,
                5.1,
                2.4,
                2.0,
                4.0,
                3.8,
                4.9,
                6.8,
                9.7,
            ]
        )

        # Calculate the model values at each time point
        def model_func(t_i):
            # Annual cycle terms (fixed period of 12 months)
            annual_cos = b2 * jnp.cos(2.0 * jnp.pi * t_i / 12.0)
            annual_sin = b3 * jnp.sin(2.0 * jnp.pi * t_i / 12.0)

            # First variable period terms
            period1_cos = b5 * jnp.cos(2.0 * jnp.pi * t_i / b4)
            period1_sin = b6 * jnp.sin(2.0 * jnp.pi * t_i / b4)

            # Second variable period terms
            period2_cos = b8 * jnp.cos(2.0 * jnp.pi * t_i / b7)
            period2_sin = b9 * jnp.sin(2.0 * jnp.pi * t_i / b7)

            # Combine all terms
            return (
                b1
                + annual_cos
                + annual_sin
                + period1_cos
                + period1_sin
                + period2_cos
                + period2_sin
            )

        # Calculate model predictions for all time points at once
        predictions = jax.vmap(model_func)(t)

        # Compute residuals
        residuals = predictions - enso_data

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Two starting points from the SIF file
        if self.start_point == 0:
            # Starting point 1
            return jnp.array([11.0, 3.0, 0.5, 40.0, -0.7, -1.3, 25.0, -0.3, 1.4])
        else:  # self.start_point == 1
            # Starting point 2
            return jnp.array([10.0, 3.0, 0.5, 44.0, -1.5, 0.5, 26.0, -0.1, 1.5])

    def args(self):
        return None

    def expected_result(self):
        # Should use certified values from NIST
        return None

    def expected_objective_value(self):
        # Should use certified minimum value from NIST
        return None
