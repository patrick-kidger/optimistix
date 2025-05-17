import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


class ECKERLE4LS(AbstractUnconstrainedMinimisation, strict=True):
    """ECKERLE4LS - Nonlinear Least-Squares problem (NIST dataset).

    This problem involves a nonlinear least squares fit to a Gaussian peak function,
    arising from a circular interference transmittance study.

    Source: Problem 7 from
    NIST nonlinear least squares test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    SIF input: Ph. Toint, April 1997.

    Classification: SUR2-MN-3-0
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

        # Extract parameters
        b1, b2, b3 = y

        # NIST dataset: x and y values
        x_data = jnp.array(
            [
                400.0,
                405.0,
                410.0,
                415.0,
                420.0,
                425.0,
                430.0,
                435.0,
                436.5,
                438.0,
                439.5,
                441.0,
                442.5,
                444.0,
                445.5,
                447.0,
                448.5,
                450.0,
                451.5,
                453.0,
                454.5,
                456.0,
                457.5,
                459.0,
                460.5,
                462.0,
                463.5,
                465.0,
                470.0,
                475.0,
                480.0,
                485.0,
                490.0,
                495.0,
                500.0,
            ]
        )

        y_data = jnp.array(
            [
                1.575e-1,
                1.699e-1,
                2.350e-1,
                3.102e-1,
                4.917e-1,
                8.710e-1,
                1.718e0,
                3.682e0,
                4.944e0,
                6.637e0,
                8.796e0,
                1.168e1,
                1.484e1,
                1.739e1,
                1.710e1,
                1.342e1,
                7.600e0,
                2.245e0,
                1.120e0,
                8.178e-1,
                6.615e-1,
                5.880e-1,
                4.401e-1,
                3.431e-1,
                2.400e-1,
                1.909e-1,
                1.750e-1,
                1.381e-1,
                9.576e-2,
                7.274e-2,
                6.907e-2,
                6.012e-2,
                5.629e-2,
                4.992e-2,
                4.670e-2,
            ]
        )

        # Model function: b1/b2 * exp[-0.5*((x-b3)/b2)^2]
        def model_func(x):
            exponent = -0.5 * ((x - b3) / b2) ** 2
            return (b1 / b2) * jnp.exp(exponent)

        # Calculate model predictions for all x values at once
        predictions = jax.vmap(model_func)(x_data)

        # Compute residuals
        residuals = predictions - y_data

        # Return the sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Two starting points from the SIF file
        if self.start_point == 0:
            # Starting point 1: b1 = 1.0, b2 = 10.0, b3 = 500.0
            return jnp.array([1.0, 10.0, 500.0])
        else:  # self.start_point == 1
            # Starting point 2: b1 = 1.5, b2 = 5.0, b3 = 450.0
            return jnp.array([1.5, 5.0, 450.0])

    def args(self):
        return None

    def expected_result(self):
        # Should use certified values from NIST
        return None

    def expected_objective_value(self):
        # Should use certified minimum value from NIST
        return None
