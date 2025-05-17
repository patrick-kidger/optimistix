import jax
import jax.numpy as jnp
import jax.scipy.special as jss

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CERI651ALS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651A given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [36844.7449265, 37300.5256846]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651A.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    n: int = 7  # Number of variables
    m: int = 61  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file (lines 49-109)
        x_data = jnp.array(
            [
                36850.62500,
                36858.00000,
                36865.37500,
                36872.75000,
                36880.12500,
                36887.50000,
                36894.87500,
                36902.25000,
                36909.62500,
                36917.00000,
                36924.37500,
                36931.75000,
                36939.12500,
                36946.50000,
                36953.87500,
                36961.26563,
                36968.67188,
                36976.07813,
                36983.48438,
                36990.89063,
                36998.29688,
                37005.70313,
                37013.10938,
                37020.51563,
                37027.92188,
                37035.32813,
                37042.73438,
                37050.14063,
                37057.54688,
                37064.95313,
                37072.35938,
                37079.76563,
                37087.17188,
                37094.57813,
                37101.98438,
                37109.39063,
                37116.81250,
                37124.25000,
                37131.68750,
                37139.12500,
                37146.56250,
                37154.00000,
                37161.43750,
                37168.87500,
                37176.31250,
                37183.75000,
                37191.18750,
                37198.62500,
                37206.06250,
                37213.50000,
                37220.93750,
                37228.37500,
                37235.81250,
                37243.25000,
                37250.68750,
                37258.12500,
                37265.56250,
                37273.01563,
                37280.48438,
                37287.95313,
                37295.42188,
            ]
        )

        # Y data points from the SIF file (lines 111-171)
        y_data = jnp.array(
            [
                0.00000000,
                1.96083316,
                2.94124974,
                0.98041658,
                5.88249948,
                1.96083316,
                3.92166632,
                3.92166632,
                3.92166632,
                4.90208290,
                2.94124974,
                14.70624870,
                15.68666528,
                21.56916476,
                41.17749637,
                64.70749429,
                108.82624040,
                132.35623832,
                173.53373469,
                186.27915023,
                224.51539686,
                269.61455955,
                256.86914400,
                268.63414297,
                293.14455747,
                277.45789219,
                211.76998132,
                210.78956474,
                176.47498443,
                151.96456993,
                126.47373884,
                80.39415957,
                95.10040828,
                71.57041035,
                65.68791087,
                37.25583005,
                40.19707979,
                25.49083108,
                22.54958134,
                26.47124766,
                19.60833160,
                20.58874818,
                14.70624870,
                11.76499896,
                6.86291606,
                4.90208290,
                1.96083316,
                6.86291606,
                8.82374922,
                0.98041658,
                1.96083316,
                3.92166632,
                5.88249948,
                7.84333264,
                3.92166632,
                3.92166632,
                3.92166632,
                2.94124974,
                0.98041658,
                0.98041658,
                2.94124974,
            ]
        )

        # Error data points from the SIF file (lines 173-233)
        e_data = jnp.array(
            [
                1.00000000,
                1.41421356,
                1.73205081,
                1.00000000,
                2.44948974,
                1.41421356,
                2.00000000,
                2.00000000,
                2.00000000,
                2.23606798,
                1.73205081,
                3.87298335,
                4.00000000,
                4.69041576,
                6.48074070,
                8.12403840,
                0.53565375,
                1.61895004,
                3.30413470,
                3.78404875,
                5.13274595,
                6.58312395,
                6.18641406,
                6.55294536,
                7.29161647,
                6.82260384,
                4.69693846,
                4.66287830,
                3.41640786,
                2.44989960,
                1.35781669,
                9.05538514,
                9.84885780,
                8.54400375,
                8.18535277,
                6.16441400,
                6.40312424,
                5.09901951,
                4.79583152,
                5.19615242,
                4.47213595,
                4.58257569,
                3.87298335,
                3.46410162,
                2.64575131,
                2.23606798,
                1.41421356,
                2.64575131,
                3.00000000,
                1.00000000,
                1.41421356,
                2.00000000,
                2.44948974,
                2.82842712,
                2.00000000,
                2.00000000,
                2.00000000,
                1.73205081,
                1.00000000,
                1.00000000,
                1.73205081,
            ]
        )

        # Define helper function for the erfc_scaled function
        def erfc_scaled(z):
            # erfc_scaled(z) = exp(z^2) * erfc(z)
            # erfc(z) = 1 - erf(z)
            # First, compute z²
            z2 = z * z
            # Then compute exp(z²) * erfc(z)
            return jnp.exp(z2) * (1.0 - jss.erf(z))

        # Weights for weighted least squares (1/error)
        weights = 1.0 / e_data

        # Define function to compute model value for a single x
        def compute_model(x):
            # Linear background term
            background = c + l * x

            # Difference term
            diff = x - x0

            # Common term in the back-to-back exponential
            prefactor = i * a * b / (2.0 * (a + b))

            # Compute the arguments for the erfc function
            z = diff / s
            ac = jnp.sqrt(0.5) * (a * s + diff / s)
            bc = jnp.sqrt(0.5) * (b * s + diff / s)

            # Compute the terms with the erfc function
            term1 = jnp.exp(-0.5 * z**2) * erfc_scaled(ac)
            term2 = jnp.exp(-0.5 * z**2) * erfc_scaled(bc)

            # Full back-to-back exponential
            b2b = prefactor * (term1 + term2)

            return background + b2b

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute weighted residuals and return sum of squares
        residuals = weights * (y_pred - y_data)
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([0.0, 0.0, 1.0, 0.05, 26061.4, 38.7105, 37027.1])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CERI651BLS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651B given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [26047.3026604, 26393.719109]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651B.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    n: int = 7  # Number of variables
    m: int = 66  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                26052.42188,
                26057.64063,
                26062.85938,
                26068.07813,
                26073.29688,
                26078.51563,
                26083.73438,
                26088.95313,
                26094.17188,
                26099.39063,
                26104.60938,
                26109.82813,
                26115.04688,
                26120.26563,
                26125.48438,
                26130.70313,
                26135.92188,
                26141.14063,
                26146.35938,
                26151.57813,
                26156.79688,
                26162.01563,
                26167.23438,
                26172.45313,
                26177.68750,
                26182.93750,
                26188.18750,
                26193.43750,
                26198.68750,
                26203.93750,
                26209.18750,
                26214.43750,
                26219.68750,
                26224.93750,
                26230.18750,
                26235.43750,
                26240.68750,
                26245.93750,
                26251.18750,
                26256.43750,
                26261.68750,
                26266.93750,
                26272.18750,
                26277.43750,
                26282.68750,
                26287.93750,
                26293.18750,
                26298.43750,
                26303.68750,
                26308.93750,
                26314.18750,
                26319.43750,
                26324.68750,
                26329.93750,
                26335.18750,
                26340.43750,
                26345.68750,
                26350.93750,
                26356.18750,
                26361.43750,
                26366.68750,
                26371.93750,
                26377.18750,
                26382.43750,
                26387.68750,
                26392.93750,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                3.92166632,
                0.98041658,
                3.92166632,
                10.78458290,
                10.78458290,
                7.84333264,
                5.88249948,
                4.90208290,
                6.86291606,
                5.88249948,
                6.86291606,
                12.74541264,
                10.78458290,
                14.70624870,
                23.53000100,
                43.13833262,
                59.80500099,
                95.10040828,
                136.27790466,
                182.35748392,
                243.14289583,
                293.14455747,
                356.87121712,
                366.67538600,
                377.46080514,
                375.49997198,
                340.20455810,
                305.89039448,
                242.16247925,
                196.08289998,
                153.92539701,
                115.68915039,
                93.13956904,
                79.41373691,
                59.80500099,
                43.13833262,
                33.33416374,
                27.45166426,
                27.45166426,
                19.60833160,
                19.60833160,
                19.60833160,
                14.70624870,
                12.74541264,
                4.90208290,
                4.90208290,
                7.84333264,
                8.82374922,
                7.84333264,
                2.94124974,
                1.96083316,
                0.00000000,
                3.92166632,
                1.96083316,
                0.98041658,
                1.96083316,
                0.98041658,
                1.96083316,
                3.92166632,
                1.96083316,
                0.98041658,
                0.00000000,
                1.96083316,
                1.96083316,
                0.00000000,
                0.98041658,
            ]
        )

        # Error data points
        e_data = jnp.array(
            [
                2.00000000,
                1.00000000,
                2.00000000,
                3.31662479,
                3.31662479,
                2.82842712,
                2.44948974,
                2.23606798,
                2.64575131,
                2.44948974,
                2.64575131,
                3.60555128,
                3.31662479,
                3.87298335,
                4.89897949,
                6.63325406,
                7.79102166,
                9.84885780,
                1.17564839,
                3.56648984,
                5.68680301,
                7.19792938,
                9.32953269,
                9.48835280,
                9.60234826,
                9.57602225,
                9.12141420,
                8.63710459,
                6.12065331,
                3.70536146,
                2.45772777,
                7.70621754,
                9.70539501,
                9.02776669,
                7.79102166,
                6.63325406,
                5.82842712,
                5.28726358,
                5.28726358,
                4.47213595,
                4.47213595,
                4.47213595,
                3.87298335,
                3.60555128,
                2.23606798,
                2.23606798,
                2.82842712,
                3.00000000,
                2.82842712,
                1.73205081,
                1.41421356,
                0.00000000,
                2.00000000,
                1.41421356,
                1.00000000,
                1.41421356,
                1.00000000,
                1.41421356,
                2.00000000,
                1.41421356,
                1.00000000,
                0.00000000,
                1.41421356,
                1.41421356,
                0.00000000,
                1.00000000,
            ]
        )

        # Define helper function for the erfc_scaled function
        def erfc_scaled(z):
            # erfc_scaled(z) = exp(z^2) * erfc(z)
            # erfc(z) = 1 - erf(z)
            # First, compute z²
            z2 = z * z
            # Then compute exp(z²) * erfc(z)
            return jnp.exp(z2) * (1.0 - jss.erf(z))

        # Weights for weighted least squares (1/error)
        weights = 1.0 / e_data

        # Define function to compute model value for a single x
        def compute_model(x):
            # Linear background term
            background = c + l * x

            # Difference term
            diff = x - x0

            # Common term in the back-to-back exponential
            prefactor = i * a * b / (2.0 * (a + b))

            # Compute the arguments for the erfc function
            z = diff / s
            ac = jnp.sqrt(0.5) * (a * s + diff / s)
            bc = jnp.sqrt(0.5) * (b * s + diff / s)

            # Compute the terms with the erfc function
            term1 = jnp.exp(-0.5 * z**2) * erfc_scaled(ac)
            term2 = jnp.exp(-0.5 * z**2) * erfc_scaled(bc)

            # Full back-to-back exponential
            b2b = prefactor * (term1 + term2)

            return background + b2b

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute weighted residuals and return sum of squares
        residuals = weights * (y_pred - y_data)
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values similar to CERI651ALS
        return jnp.array([0.0, 0.0, 1.0, 0.05, 26061.4, 38.7105, 26227.1])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CERI651CLS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651C given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [23919.5789114, 24189.3183142]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651C.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    n: int = 7  # Number of variables
    m: int = 56  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                23920.10938,
                23924.89063,
                23929.67188,
                23934.45313,
                23939.23438,
                23944.01563,
                23948.79688,
                23953.57813,
                23958.35938,
                23963.14063,
                23967.92188,
                23972.70313,
                23977.48438,
                23982.26563,
                23987.06250,
                23991.87500,
                23996.68750,
                24001.50000,
                24006.31250,
                24011.12500,
                24015.93750,
                24020.75000,
                24025.56250,
                24030.37500,
                24035.18750,
                24040.00000,
                24044.81250,
                24049.62500,
                24054.43750,
                24059.25000,
                24064.06250,
                24068.87500,
                24073.68750,
                24078.50000,
                24083.31250,
                24088.12500,
                24092.93750,
                24097.75000,
                24102.56250,
                24107.37500,
                24112.18750,
                24117.00000,
                24121.81250,
                24126.62500,
                24131.43750,
                24136.25000,
                24141.06250,
                24145.89063,
                24150.73438,
                24155.57813,
                24160.42188,
                24165.26563,
                24170.10938,
                24174.95313,
                24179.79688,
                24184.64063,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                0.00000000,
                0.98041658,
                1.96083316,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                3.92166632,
                0.98041658,
                0.00000000,
                0.98041658,
                2.94124974,
                1.96083316,
                0.98041658,
                2.94124974,
                8.82374922,
                5.88249948,
                6.86291606,
                8.82374922,
                11.76499896,
                12.74541554,
                6.86291606,
                8.82374922,
                12.74541554,
                13.72583212,
                8.82374922,
                12.74541554,
                19.60833160,
                4.90208290,
                2.94124974,
                1.96083316,
                3.92166632,
                3.92166632,
                5.88249948,
                2.94124974,
                4.90208290,
                6.86291606,
                2.94124974,
                1.96083316,
                0.00000000,
                1.96083316,
                2.94124974,
                1.96083316,
                1.96083316,
                1.96083316,
                3.92166632,
                0.00000000,
                0.00000000,
                3.92166632,
                2.94124974,
                1.96083316,
                0.00000000,
                1.96083316,
                0.00000000,
                0.98041658,
                0.98041658,
            ]
        )

        # Error data points
        e_data = jnp.array(
            [
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                2.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.73205081,
                1.41421356,
                1.00000000,
                1.73205081,
                3.00000000,
                2.44948974,
                2.64575131,
                3.00000000,
                3.46410162,
                3.60555128,
                2.64575131,
                3.00000000,
                3.60555128,
                3.74165739,
                3.00000000,
                3.60555128,
                4.47213595,
                2.23606798,
                1.73205081,
                1.41421356,
                2.00000000,
                2.00000000,
                2.44948974,
                1.73205081,
                2.23606798,
                2.64575131,
                1.73205081,
                1.41421356,
                1.00000000,
                1.41421356,
                1.73205081,
                1.41421356,
                1.41421356,
                1.41421356,
                2.00000000,
                1.00000000,
                1.00000000,
                2.00000000,
                1.73205081,
                1.41421356,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
            ]
        )

        # Define helper function for the erfc_scaled function
        def erfc_scaled(z):
            # erfc_scaled(z) = exp(z^2) * erfc(z)
            # erfc(z) = 1 - erf(z)
            # First, compute z²
            z2 = z * z
            # Then compute exp(z²) * erfc(z)
            return jnp.exp(z2) * (1.0 - jss.erf(z))

        # Weights for weighted least squares (1/error)
        weights = 1.0 / e_data

        # Define function to compute model value for a single x
        def compute_model(x):
            # Linear background term
            background = c + l * x

            # Difference term
            diff = x - x0

            # Common term in the back-to-back exponential
            prefactor = i * a * b / (2.0 * (a + b))

            # Compute the arguments for the erfc function
            z = diff / s
            ac = jnp.sqrt(0.5) * (a * s + diff / s)
            bc = jnp.sqrt(0.5) * (b * s + diff / s)

            # Compute the terms with the erfc function
            term1 = jnp.exp(-0.5 * z**2) * erfc_scaled(ac)
            term2 = jnp.exp(-0.5 * z**2) * erfc_scaled(bc)

            # Full back-to-back exponential
            b2b = prefactor * (term1 + term2)

            return background + b2b

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute weighted residuals and return sum of squares
        residuals = weights * (y_pred - y_data)
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from the SIF file (START3)
        return jnp.array([0.0, 0.0, 1.0, 0.05, 597.076, 22.9096, 24027.5])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CERI651DLS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651D given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [12986.356148, 13161.356148]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651D.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    n: int = 7  # Number of variables
    m: int = 67  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                12987.48438,
                12990.07813,
                12992.67188,
                12995.26563,
                12997.85938,
                13000.45313,
                13003.04688,
                13005.64063,
                13008.23438,
                13010.82813,
                13013.42188,
                13016.01563,
                13018.60938,
                13021.20313,
                13023.79688,
                13026.39063,
                13028.98438,
                13031.57813,
                13034.17188,
                13036.76563,
                13039.35938,
                13041.95313,
                13044.54688,
                13047.14063,
                13049.75000,
                13052.37500,
                13055.00000,
                13057.62500,
                13060.25000,
                13062.87500,
                13065.50000,
                13068.12500,
                13070.75000,
                13073.37500,
                13076.00000,
                13078.62500,
                13081.25000,
                13083.87500,
                13086.50000,
                13089.12500,
                13091.75000,
                13094.37500,
                13097.00000,
                13099.62500,
                13102.25000,
                13104.87500,
                13107.50000,
                13110.12500,
                13112.75000,
                13115.37500,
                13118.00000,
                13120.62500,
                13123.25000,
                13125.87500,
                13128.50000,
                13131.12500,
                13133.75000,
                13136.37500,
                13139.00000,
                13141.62500,
                13144.25000,
                13146.87500,
                13149.50000,
                13152.12500,
                13154.75000,
                13157.37500,
                13160.00000,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                1.96083316,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                4.90208290,
                0.98041658,
                0.98041658,
                0.98041658,
                3.92166632,
                1.96083316,
                1.96083316,
                0.98041658,
                1.96083316,
                1.96083316,
                1.96083316,
                0.98041658,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.98041658,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                1.96083316,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
            ]
        )

        # Error data points from the SIF file
        e_data = jnp.array(
            [
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                2.23606798,
                1.00000000,
                1.00000000,
                1.00000000,
                2.00000000,
                1.41421356,
                1.41421356,
                1.00000000,
                1.41421356,
                1.41421356,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
            ]
        )

        # Define helper function for the erfc_scaled function
        def erfc_scaled(z):
            # erfc_scaled(z) = exp(z^2) * erfc(z)
            # erfc(z) = 1 - erf(z)
            # First, compute z²
            z2 = z * z
            # Then compute exp(z²) * erfc(z)
            return jnp.exp(z2) * (1.0 - jss.erf(z))

        # Weights for weighted least squares (1/error)
        weights = 1.0 / e_data

        # Define function to compute model value for a single x
        def compute_model(x):
            # Linear background term
            background = c + l * x

            # Difference term
            diff = x - x0

            # Common term in the back-to-back exponential
            prefactor = i * a * b / (2.0 * (a + b))

            # Compute the arguments for the erfc function
            z = diff / s
            ac = jnp.sqrt(0.5) * (a * s + diff / s)
            bc = jnp.sqrt(0.5) * (b * s + diff / s)

            # Compute the terms with the erfc function
            term1 = jnp.exp(-0.5 * z**2) * erfc_scaled(ac)
            term2 = jnp.exp(-0.5 * z**2) * erfc_scaled(bc)

            # Full back-to-back exponential
            b2b = prefactor * (term1 + term2)

            return background + b2b

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute weighted residuals and return sum of squares
        residuals = weights * (y_pred - y_data)
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from the SIF file (START4)
        return jnp.array([0.0, 0.0, 1.0, 0.05, 15.1595, 8.0, 13072.9])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CERI651ELS(AbstractUnconstrainedMinimisation):
    """ISIS Data fitting problem CERI651E given as an inconsistent set of
    nonlinear equations.

    Fit: y = c + l * x + I*A*B/2(A+B) *
         [ exp( A*[A*S^2+2(x-X0)]/2) * erfc( A*S^2+(x-X0)/S*sqrt(2) ) +
           exp( B*[B*S^2+2(x-X0)]/2) * erfc( B*S^2+(x-X0)/S*sqrt(2) ) ]

    Source: fit to a sum of a linear background and a back-to-back exponential
    using data enginx_ceria193749_spectrum_number_651_vana_corrected-0
    from Mantid (http://www.mantidproject.org)

    subset X in [13556.2988352, 13731.2988352]

    SIF input: Nick Gould and Tyrone Rees, Mar 2016
    Least-squares version of CERI651E.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-MN-7-0
    """

    n: int = 7  # Number of variables
    m: int = 64  # Number of data points

    def objective(self, y, args):
        del args
        c, l, a, b, i, s, x0 = y

        # X data points from the SIF file
        x_data = jnp.array(
            [
                13558.04688,
                13560.76563,
                13563.48438,
                13566.20313,
                13568.92188,
                13571.64063,
                13574.35938,
                13577.07813,
                13579.79688,
                13582.51563,
                13585.23438,
                13587.95313,
                13590.67188,
                13593.39063,
                13596.10938,
                13598.82813,
                13601.54688,
                13604.26563,
                13606.98438,
                13609.70313,
                13612.42188,
                13615.14063,
                13617.85938,
                13620.57813,
                13623.29688,
                13626.01563,
                13628.73438,
                13631.45313,
                13634.17188,
                13636.89063,
                13639.60938,
                13642.32813,
                13645.04688,
                13647.76563,
                13650.48438,
                13653.20313,
                13655.92188,
                13658.64063,
                13661.35938,
                13664.07813,
                13666.79688,
                13669.51563,
                13672.23438,
                13674.96875,
                13677.71875,
                13680.46875,
                13683.21875,
                13685.96875,
                13688.71875,
                13691.46875,
                13694.21875,
                13696.96875,
                13699.71875,
                13702.46875,
                13705.21875,
                13707.96875,
                13710.71875,
                13713.46875,
                13716.21875,
                13718.96875,
                13721.71875,
                13724.46875,
                13727.21875,
                13729.96875,
            ]
        )

        # Y data points from the SIF file
        y_data = jnp.array(
            [
                0.00000000,
                1.96083316,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.98041658,
                0.98041658,
                1.96083316,
                1.96083316,
                4.90208290,
                0.98041658,
                1.96083316,
                0.00000000,
                1.96083316,
                0.98041658,
                5.88249948,
                0.98041658,
                1.96083316,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                1.96083316,
                0.98041658,
                0.00000000,
                0.98041658,
                0.98041658,
                0.98041658,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.98041658,
                0.00000000,
                0.00000000,
                0.00000000,
                0.98041658,
                0.98041658,
                0.98041658,
                0.00000000,
                0.98041658,
                0.00000000,
                1.96083316,
                0.00000000,
                0.00000000,
            ]
        )

        # Error data points from the SIF file
        e_data = jnp.array(
            [
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.41421356,
                2.23606798,
                1.00000000,
                1.41421356,
                1.00000000,
                1.41421356,
                1.00000000,
                2.44948974,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.41421356,
                1.00000000,
                1.00000000,
            ]
        )

        # Define helper function for the erfc_scaled function
        def erfc_scaled(z):
            # erfc_scaled(z) = exp(z^2) * erfc(z)
            # erfc(z) = 1 - erf(z)
            # First, compute z²
            z2 = z * z
            # Then compute exp(z²) * erfc(z)
            return jnp.exp(z2) * (1.0 - jss.erf(z))

        # Weights for weighted least squares (1/error)
        weights = 1.0 / e_data

        # Define function to compute model value for a single x
        def compute_model(x):
            # Linear background term
            background = c + l * x

            # Difference term
            diff = x - x0

            # Common term in the back-to-back exponential
            prefactor = i * a * b / (2.0 * (a + b))

            # Compute the arguments for the erfc function
            z = diff / s
            ac = jnp.sqrt(0.5) * (a * s + diff / s)
            bc = jnp.sqrt(0.5) * (b * s + diff / s)

            # Compute the terms with the erfc function
            term1 = jnp.exp(-0.5 * z**2) * erfc_scaled(ac)
            term2 = jnp.exp(-0.5 * z**2) * erfc_scaled(bc)

            # Full back-to-back exponential
            b2b = prefactor * (term1 + term2)

            return background + b2b

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(compute_model)(x_data)

        # Compute weighted residuals and return sum of squares
        residuals = weights * (y_pred - y_data)
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from the SIF file (START5)
        return jnp.array([0.0, 0.0, 1.0, 0.05, 17.06794, 8.0, 13642.3])

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None
