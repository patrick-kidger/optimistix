import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: claude seems to struggle adding all the data and starting points provided.
# Perhaps this is just longer than the context window it has?
# TODO: This implementation requires human review and verification against
# another CUTEst interface
class GAUSS1LS(AbstractUnconstrainedMinimisation, strict=True):
    """The GAUSS1LS function.

    NIST Data fitting problem GAUSS1.

    Fit: y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2) + e

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Rust, B., NIST (1996).

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification SUR2-MN-8-0
    """

    # Number of variables
    n: int = 8

    def objective(self, y, args):
        """Compute the objective function value.

        Args:
            y: The parameters [b1, b2, b3, b4, b5, b6, b7, b8]
            args: None

        Returns:
            The sum of squared residuals.
        """
        del args
        b1, b2, b3, b4, b5, b6, b7, b8 = y

        # Create the x data points (1 to 250)
        x = jnp.arange(1.0, 251.0)

        # Model function:
        # b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)
        term1 = b1 * jnp.exp(-b2 * x)
        term2 = b3 * jnp.exp(-((x - b4) ** 2) / (b5**2))
        term3 = b6 * jnp.exp(-((x - b7) ** 2) / (b8**2))
        model = term1 + term2 + term3

        # Actual y values from the dataset (hard-coded from the SIF file)
        y = jnp.array(
            [
                97.62227,
                97.80724,
                96.62247,
                92.59022,
                91.23869,
                95.32704,
                90.35040,
                89.46235,
                91.72520,
                89.86916,
                86.88076,
                85.94360,
                87.60686,
                86.25839,
                80.74976,
                83.03551,
                88.25837,
                82.01316,
                82.74098,
                83.30034,
                81.27850,
                81.85506,
                80.75195,
                80.09573,
                81.07633,
                78.81542,
                78.38596,
                79.93386,
                79.48474,
                79.95942,
                76.10691,
                78.39830,
                81.43060,
                82.48867,
                81.65462,
                80.84323,
                88.68663,
                84.74438,
                86.83934,
                85.97739,
                91.28509,
                97.22411,
                93.51733,
                94.10159,
                101.91760,
                98.43134,
                110.4214,
                107.6628,
                111.7288,
                116.5115,
                120.7609,
                123.9553,
                124.2437,
                130.7996,
                133.2960,
                130.7788,
                132.0565,
                138.6584,
                142.9252,
                142.7215,
                144.1249,
                147.4377,
                148.2647,
                152.0519,
                147.3863,
                149.2074,
                148.9537,
                144.5876,
                148.1226,
                148.0144,
                143.8893,
                140.9088,
                143.4434,
                139.3938,
                135.9878,
                136.3927,
                126.7262,
                124.4487,
                122.8647,
                113.8557,
                113.7037,
                106.8407,
                107.0034,
                102.46290,
                96.09296,
                94.57555,
                86.98824,
                84.90154,
                81.18023,
                76.40117,
                67.09200,
                72.67155,
                68.10848,
                67.99088,
                63.34094,
                60.55253,
                56.18687,
                53.64482,
                53.70307,
                48.07893,
                42.21258,
                45.65181,
                41.69728,
                41.24946,
                39.21349,
                37.71696,
                36.68395,
                37.30393,
                37.43277,
                37.45012,
                32.64648,
                31.84347,
                31.39951,
                26.68912,
                32.25323,
                27.61008,
                33.58649,
                28.10714,
                30.26428,
                28.01648,
                29.11021,
                23.02099,
                25.65091,
                28.50295,
                25.23701,
                26.13828,
                33.53260,
                29.25195,
                27.09847,
                26.52999,
                25.52401,
                26.69218,
                24.55269,
                27.71763,
                25.20297,
                25.61483,
                25.06893,
                27.63930,
                24.94851,
                25.86806,
                22.48183,
                26.90045,
                25.39919,
                17.90614,
                23.76039,
                25.89689,
                27.64231,
                22.86101,
                26.47003,
                23.72888,
                27.54334,
                30.52683,
                28.07261,
                34.92815,
                28.29194,
                34.19161,
                35.41207,
                37.09336,
                40.98330,
                39.53923,
                47.80123,
                47.46305,
                51.04166,
                54.58065,
                57.53001,
                61.42089,
                62.79032,
                68.51455,
                70.23053,
                74.42776,
                76.59911,
                81.62053,
                83.42208,
                79.17451,
                88.56985,
                85.66525,
                86.55502,
                90.65907,
                84.27290,
                85.72220,
                83.10702,
                82.16884,
                80.42568,
                78.15692,
                79.79691,
                77.84378,
                74.50327,
                71.57289,
                65.88031,
                65.01385,
                60.19582,
                59.66726,
                52.95478,
                53.87792,
                44.91274,
                41.09909,
                41.68018,
                34.53379,
                34.86419,
                33.14787,
                29.58864,
                27.29462,
                21.91439,
                19.08159,
                24.90290,
                19.82341,
                16.75551,
                18.24558,
                17.23549,
                16.34934,
                13.71285,
                14.75676,
                13.97169,
                12.42867,
                14.35519,
                7.703309,
                10.234410,
                11.78315,
                13.87768,
                4.535700,
                10.059280,
                8.424824,
                10.533120,
                9.602255,
                7.877514,
                6.258121,
                8.899865,
                7.877754,
                12.51191,
                10.66205,
                6.035400,
                6.790655,
                8.783535,
                4.600288,
                8.400915,
                7.216561,
                10.017410,
                7.331278,
                6.527863,
                2.842001,
                10.325070,
                4.790995,
                8.377101,
                6.264445,
                2.706213,
                8.362329,
                8.983658,
                3.362571,
                1.182746,
                4.875359,
            ]
        )

        # Sum of squared residuals (least squares objective)
        residuals = model - y
        return jnp.sum(residuals**2)

    def y0(self):
        """Return the starting point from the SIF file."""
        # START1 values from SIF file
        return jnp.array([97.0, 0.009, 100.0, 65.0, 20.0, 70.0, 178.0, 16.5])

    def args(self):
        """Return None as no additional args are needed."""
        return None

    def expected_result(self):
        """Return None as the exact solution is not specified in the SIF file."""
        # The problem doesn't specify the exact minimum point
        return None

    def expected_objective_value(self):
        # The problem doesn't specify the minimum objective value in the SIF file
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class GAUSS2LS(AbstractUnconstrainedMinimisation, strict=True):
    """The GAUSS2LS function.

    NIST Data fitting problem GAUSS2.

    Fit: y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2) + e

    Similar to GAUSS1LS but with different data.

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Rust, B., NIST (1996).

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification SUR2-MN-8-0
    """

    # This problem has the same model as GAUSS1LS but different data
    # Implementation would be similar to GAUSS1LS but with different y values

    # The actual implementation would need to be completed after examining
    # the GAUSS2LS.SIF file
    n: int = 8

    def objective(self, y, args):
        """Placeholder for GAUSS2LS objective function."""
        # This would need to be implemented based on the GAUSS2LS.SIF file
        return jnp.sum(y**2)  # Placeholder

    def y0(self):
        """Placeholder for GAUSS2LS starting point."""
        # This would need to be implemented based on the GAUSS2LS.SIF file
        return jnp.zeros(self.n)  # Placeholder

    def args(self):
        """Return None as no additional args are needed."""
        return None

    def expected_result(self):
        """Return None as the exact solution is not specified."""
        return None

    def expected_objective_value(self):
        # The minimum objective value is not specified
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class GAUSS3LS(AbstractUnconstrainedMinimisation, strict=True):
    """The GAUSS3LS function.

    NIST Data fitting problem GAUSS3.

    Fit: y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2) + e

    Similar to GAUSS1LS but with different data.

    Source: Problem from the NIST nonlinear regression test set
    http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Reference: Rust, B., NIST (1996).

    SIF input: Nick Gould and Tyrone Rees, Oct 2015

    Classification SUR2-MN-8-0
    """

    # This problem has the same model as GAUSS1LS but different data
    # Implementation would be similar to GAUSS1LS but with different y values

    # The actual implementation would need to be completed after examining
    # the GAUSS3LS.SIF file
    n: int = 8

    def objective(self, y, args):
        """Placeholder for GAUSS3LS objective function."""
        # This would need to be implemented based on the GAUSS3LS.SIF file
        return jnp.sum(y**2)  # Placeholder

    def y0(self):
        """Placeholder for GAUSS3LS starting point."""
        # This would need to be implemented based on the GAUSS3LS.SIF file
        return jnp.zeros(self.n)  # Placeholder

    def args(self):
        """Return None as no additional args are needed."""
        return None

    def expected_result(self):
        """Return None as the exact solution is not specified."""
        return None

    def expected_objective_value(self):
        # The minimum objective value is not specified
        return None
