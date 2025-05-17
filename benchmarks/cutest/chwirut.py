import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CHWIRUT1LS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem CHWIRUT1.

    The objective is to minimize a least-squares function with the model:
    y = exp[-b1*x]/(b2+b3*x) + e

    where b1, b2, b3 are the parameters to be determined and e represents
    the error term.

    Source: Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Chwirut, D., NIST (197?).
      Ultrasonic Reference Block Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: SUR2-MN-3-0
    """

    n: int = 3  # Number of variables
    m: int = 214  # Number of data points

    def objective(self, y, args):
        del args
        b1, b2, b3 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
                1.25,
                1.75,
                2.25,
                1.75,
                2.25,
                2.75,
                3.25,
                3.75,
                4.25,
                4.75,
                5.25,
                5.75,
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
                1.25,
                1.75,
                2.25,
                1.75,
                2.25,
                2.75,
                3.25,
                3.75,
                4.25,
                4.75,
                5.25,
                5.75,
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
                1.25,
                1.75,
                2.25,
                1.75,
                2.25,
                2.75,
                3.25,
                3.75,
                4.25,
                4.75,
                5.25,
                5.75,
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
                1.25,
                1.75,
                2.25,
                1.75,
                2.25,
                2.75,
                3.25,
                3.75,
                4.25,
                4.75,
                5.25,
                5.75,
                0.5,
                0.75,
                1.5,
                3.0,
                3.0,
                3.0,
                6.0,
                0.5,
                0.75,
                1.5,
                3.0,
                3.0,
                3.0,
                6.0,
                0.5,
                0.75,
                1.5,
                3.0,
                3.0,
                3.0,
                6.0,
                0.5,
                0.75,
                1.5,
                3.0,
                6.0,
                3.0,
                3.0,
                6.0,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
                1.25,
                2.25,
                2.25,
                2.75,
                3.25,
                3.75,
                4.25,
                4.75,
                5.25,
                5.75,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.5,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.5,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.5,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                3.0,
                0.5,
                0.75,
                1.5,
                3.0,
                6.0,
                3.0,
                6.0,
                3.0,
                3.0,
                3.0,
                1.75,
                1.75,
                0.5,
                0.75,
                1.75,
                1.75,
                2.75,
                3.75,
                1.75,
                1.75,
                0.5,
                0.75,
                2.75,
                3.75,
                1.75,
                1.75,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                92.9,
                78.7,
                64.2,
                64.9,
                57.1,
                43.3,
                31.1,
                23.6,
                31.05,
                23.775,
                17.7375,
                13.8,
                11.5875,
                9.4125,
                7.725,
                7.35,
                8.025,
                90.6,
                76.9,
                71.6,
                63.6,
                54.0,
                39.2,
                29.3,
                21.4,
                29.175,
                22.125,
                17.5125,
                14.25,
                9.45,
                9.15,
                7.9125,
                8.475,
                6.1125,
                80.0,
                79.0,
                63.8,
                57.2,
                53.2,
                42.5,
                26.8,
                20.4,
                26.85,
                21.0,
                16.4625,
                12.525,
                10.5375,
                8.5875,
                7.125,
                6.1125,
                5.9625,
                74.1,
                67.3,
                60.8,
                55.5,
                50.3,
                41.0,
                29.4,
                20.4,
                29.3625,
                21.15,
                16.7625,
                13.2,
                10.875,
                8.175,
                7.35,
                5.9625,
                5.625,
                81.5,
                62.4,
                32.5,
                12.41,
                13.12,
                15.56,
                5.63,
                78.0,
                59.9,
                33.2,
                13.84,
                12.75,
                14.62,
                3.94,
                76.8,
                61.0,
                32.9,
                13.87,
                11.81,
                13.31,
                5.44,
                78.0,
                63.5,
                33.8,
                12.56,
                5.63,
                12.75,
                13.12,
                5.44,
                76.8,
                60.0,
                47.8,
                32.0,
                22.2,
                22.57,
                18.82,
                13.95,
                11.25,
                9.0,
                6.67,
                75.8,
                62.0,
                48.8,
                35.2,
                20.0,
                20.32,
                19.31,
                12.75,
                10.42,
                7.31,
                7.42,
                70.5,
                59.5,
                48.5,
                35.8,
                21.0,
                21.67,
                21.0,
                15.64,
                8.17,
                8.55,
                10.12,
                78.0,
                66.0,
                62.0,
                58.0,
                47.7,
                37.8,
                20.2,
                21.07,
                13.87,
                9.67,
                7.76,
                5.44,
                4.87,
                4.01,
                3.75,
                24.19,
                25.76,
                18.07,
                11.81,
                12.07,
                16.12,
                70.8,
                54.7,
                48.0,
                39.8,
                29.8,
                23.7,
                29.62,
                23.81,
                17.7,
                11.55,
                12.07,
                8.74,
                80.7,
                61.3,
                47.5,
                29.0,
                24.0,
                17.7,
                24.56,
                18.67,
                16.24,
                8.74,
                7.87,
                8.51,
                66.7,
                59.2,
                40.8,
                30.7,
                25.7,
                16.3,
                25.99,
                16.95,
                13.35,
                8.62,
                7.2,
                6.64,
                13.69,
                81.0,
                64.5,
                35.5,
                13.31,
                4.87,
                12.94,
                5.06,
                15.19,
                14.62,
                15.64,
                25.5,
                25.95,
                81.7,
                61.6,
                29.8,
                29.81,
                17.17,
                10.39,
                28.4,
                28.69,
                81.3,
                60.9,
                16.65,
                10.05,
                28.9,
                28.95,
            ]
        )

        # Define function to compute model value for a single x
        def compute_model(x):
            # Model: y = exp[-b1*x]/(b2+b3*x)
            numerator = jnp.exp(-b1 * x)
            denominator = b2 + b3 * x
            return numerator / denominator

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute residuals and return sum of squares
        residuals = y_pred - y_data
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([0.1, 0.01, 0.02])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not directly provided in the SIF file
        # Values from NIST: 0.1901261, 0.0061962, 0.0105
        return jnp.array([0.1901261, 0.0061962, 0.0105])

    def expected_objective_value(self):
        # Certified value from NIST: 2.3894212
        return jnp.array(2.3894212)


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CHWIRUT2LS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem CHWIRUT2.

    The objective is to minimize a least-squares function with the model:
    y = exp[-b1*x]/(b2+b3*x) + e

    where b1, b2, b3 are the parameters to be determined and e represents
    the error term.

    Source: Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Chwirut, D., NIST (197?).
      Ultrasonic Reference Block Study.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification: SUR2-MN-3-0
    """

    n: int = 3  # Number of variables
    m: int = 54  # Number of data points

    def objective(self, y, args):
        del args
        b1, b2, b3 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                0.5,
                1.0,
                1.75,
                3.75,
                5.75,
                0.875,
                2.25,
                3.25,
                5.25,
                0.75,
                1.75,
                2.75,
                4.75,
                0.625,
                1.25,
                2.25,
                4.25,
                0.5,
                3.0,
                0.75,
                3.0,
                1.5,
                6.0,
                3.0,
                6.0,
                1.5,
                3.0,
                0.5,
                2.0,
                4.0,
                0.75,
                2.0,
                5.0,
                0.75,
                2.25,
                3.75,
                5.75,
                3.0,
                0.75,
                2.5,
                4.0,
                0.75,
                2.5,
                4.0,
                0.75,
                2.5,
                4.0,
                0.5,
                6.0,
                3.0,
                0.5,
                2.75,
                0.5,
                1.75,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                92.9,
                57.1,
                31.05,
                11.5875,
                8.025,
                63.6,
                21.4,
                14.25,
                8.475,
                63.8,
                26.8,
                16.4625,
                7.125,
                67.3,
                41.0,
                21.15,
                8.175,
                81.5,
                13.12,
                59.9,
                14.62,
                32.9,
                5.44,
                12.56,
                5.44,
                32.0,
                13.95,
                75.8,
                20.0,
                10.42,
                59.5,
                21.67,
                8.55,
                62.0,
                20.2,
                7.76,
                3.75,
                11.81,
                54.7,
                23.7,
                11.55,
                61.3,
                17.7,
                8.74,
                59.2,
                16.3,
                8.62,
                81.0,
                4.87,
                14.62,
                81.7,
                17.17,
                81.3,
                28.9,
            ]
        )

        # Define function to compute model value for a single x
        def compute_model(x):
            # Model: y = exp[-b1*x]/(b2+b3*x)
            numerator = jnp.exp(-b1 * x)
            denominator = b2 + b3 * x
            return numerator / denominator

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute residuals and return sum of squares
        residuals = y_pred - y_data
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([0.1, 0.01, 0.02])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not directly provided in the SIF file
        # Values from NIST: 0.167, 0.0052, 0.011
        return jnp.array([0.167, 0.0052, 0.011])

    def expected_objective_value(self):
        # Certified value from NIST: 513.05
        return jnp.array(513.05)
