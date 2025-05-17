import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires verification against another CUTEst interface
class COATING(AbstractUnconstrainedMinimisation):
    """The MINPACK 2 Coating Thickness Standardization problem.

    This problem arises from a chemical process in which a thin film is
    applied to a stainless steel strip. The film has a complex structure
    which is a function of the process parameters. The model for the
    resulting film thickness involves 134 variables.

    Source:
    The MINPACK-2 test problem collection,
    Brett M, Averick, Richard G. Carter, Jorge J. More and Guo-iang Xue,
    Mathematics and Computer Science Division,
    Preprint MCS-P153-0692, June 1992

    SIF input: Nick Gould, Jan 2020

    Classification: SUR2-MN-134-0
    """

    n: int = 134  # Number of variables
    m: int = 252  # Number of data points

    def objective(self, y, args):
        del args

        # Constants from the SIF file
        scale1 = 4.08
        scale2 = 0.417

        # ETA1 data (lines 44-107 in the SIF file)
        eta1_data = jnp.array(
            [
                0.7140,
                0.7169,
                0.7232,
                0.7151,
                0.6848,
                0.7070,
                0.7177,
                0.7073,
                0.6734,
                0.7174,
                0.7125,
                0.6947,
                0.7121,
                0.7166,
                0.6894,
                0.6897,
                0.7024,
                0.7026,
                0.6800,
                0.6957,
                0.6987,
                0.7111,
                0.7097,
                0.6809,
                0.7139,
                0.7046,
                0.6950,
                0.7032,
                0.7019,
                0.6975,
                0.6955,
                0.7056,
                0.6965,
                0.6848,
                0.6995,
                0.6105,
                0.6027,
                0.6084,
                0.6081,
                0.6057,
                0.6116,
                0.6052,
                0.6136,
                0.6032,
                0.6081,
                0.6092,
                0.6122,
                0.6157,
                0.6191,
                0.6169,
                0.5483,
                0.5371,
                0.5576,
                0.5521,
                0.5495,
                0.5499,
                0.4937,
                0.5092,
                0.5433,
                0.5018,
                0.5363,
                0.4977,
                0.5296,
            ]
        )

        # ETA2 data (lines 108-171 in the SIF file)
        eta2_data = jnp.array(
            [
                5.145,
                5.241,
                5.389,
                5.211,
                5.154,
                5.105,
                5.191,
                5.013,
                5.582,
                5.208,
                5.142,
                5.284,
                5.262,
                6.838,
                6.215,
                6.817,
                6.889,
                6.732,
                6.717,
                6.468,
                6.776,
                6.574,
                6.465,
                6.090,
                6.350,
                4.255,
                4.154,
                4.211,
                4.287,
                4.104,
                4.007,
                4.261,
                4.150,
                4.040,
                4.155,
                5.086,
                5.021,
                5.040,
                5.247,
                5.125,
                5.136,
                4.949,
                5.253,
                5.154,
                5.227,
                5.120,
                5.291,
                5.294,
                5.304,
                5.209,
                5.384,
                5.490,
                5.563,
                5.532,
                5.372,
                5.423,
                7.237,
                6.944,
                6.957,
                7.138,
                7.009,
                7.074,
                7.046,
            ]
        )

        # Y data (lines 172-298 in the SIF file)
        y_data = jnp.array(
            [
                9.3636,
                9.3512,
                9.4891,
                9.1888,
                9.3161,
                9.2585,
                9.2913,
                9.3914,
                9.4524,
                9.4995,
                9.4179,
                9.468,
                9.4799,
                11.2917,
                11.5062,
                11.4579,
                11.3977,
                11.3688,
                11.3897,
                11.3104,
                11.3882,
                11.3629,
                11.3149,
                11.2474,
                11.2507,
                8.1678,
                8.1017,
                8.3506,
                8.3651,
                8.2994,
                8.1514,
                8.2229,
                8.1027,
                8.3785,
                8.4118,
                8.0955,
                8.0613,
                8.0979,
                8.1364,
                8.1700,
                8.1684,
                8.0885,
                8.1839,
                8.1478,
                8.1827,
                8.029,
                8.1000,
                8.2579,
                8.2248,
                8.2540,
                6.8518,
                6.8547,
                6.8831,
                6.9137,
                6.8984,
                6.8888,
                8.5189,
                8.5308,
                8.5184,
                8.5222,
                8.5705,
                8.5353,
                8.5213,
                8.3158,
                8.1995,
                8.2283,
                8.1857,
                8.2738,
                8.2131,
                8.2613,
                8.2315,
                8.2078,
                8.2996,
                8.3026,
                8.0995,
                8.2990,
                9.6753,
                9.6687,
                9.5704,
                9.5435,
                9.6780,
                9.7668,
                9.7827,
                9.7844,
                9.7011,
                9.8006,
                9.7610,
                9.7813,
                7.3073,
                7.2572,
                7.4686,
                7.3659,
                7.3587,
                7.3132,
                7.3542,
                7.2339,
                7.4375,
                7.4022,
                10.7914,
                10.6554,
                10.7359,
                10.7583,
                10.7735,
                10.7907,
                10.6465,
                10.6994,
                10.7756,
                10.7402,
                10.6800,
                10.7000,
                10.8160,
                10.6921,
                10.8677,
                12.3495,
                12.4424,
                12.4303,
                12.5086,
                12.4513,
                12.4625,
                16.2290,
                16.2781,
                16.2082,
                16.2715,
                16.2464,
                16.1626,
                16.1568,
            ]
        )

        # Number of data points divided by 4
        m4 = len(eta1_data)

        # Define function to compute residuals for a given index i (0-based)
        def compute_residuals(i):
            # Adjust for 0-based indexing
            i_1 = i  # First quarter index
            i_2 = i + m4  # Second quarter index
            i_3 = i + 2 * m4  # Third quarter index
            i_4 = i + 3 * m4  # Fourth quarter index

            # Get the corresponding data
            eta1 = eta1_data[i]
            eta2 = eta2_data[i]

            # Compute intermediate products
            i_p8 = i + 8  # IP8 in SIF
            i2_p8 = i + m4 + 8  # I2P8 in SIF

            # First quarter residual (lines 307-316)
            f_i1 = (
                y[0]
                + y[1] * eta1
                + y[2] * eta2
                + y[3] * eta1 * eta2
                + y[1] * y[i_p8] * eta2
                + y[2] * y[i2_p8] * eta1
                + y[3] * y[i_p8]
                + y[3] * y[i2_p8]
                + y[3] * y[i_p8] * y[i2_p8]
            )

            # Second quarter residual (lines 318-321)
            f_i2 = (
                y[4]
                + y[5] * eta1
                + y[6] * eta2
                + y[7] * eta1 * eta2
                + y[5] * y[i_p8] * eta2
                + y[6] * y[i2_p8] * eta1
                + y[7] * y[i_p8]
                + y[7] * y[i2_p8]
                + y[7] * y[i_p8] * y[i2_p8]
            )

            # Third quarter residual (lines 323-326)
            f_i3 = y[i_p8] * scale1

            # Fourth quarter residual (lines 328-330)
            f_i4 = y[i2_p8] * scale2

            # Compute residuals
            r_i1 = f_i1 - y_data[i_1]
            r_i2 = f_i2 - y_data[i_2]
            r_i3 = f_i3 - y_data[i_3]
            r_i4 = f_i4 - y_data[i_4]

            return jnp.array([r_i1, r_i2, r_i3, r_i4])

        # Compute residuals for all indices
        all_residuals = jax.vmap(compute_residuals)(jnp.arange(m4))

        # Flatten and sum squares
        all_residuals = all_residuals.reshape(-1)
        return jnp.sum(all_residuals**2)

    def y0(self):
        # Initial values from SIF file (all ones)
        return jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not provided in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file doesn't specify the optimal objective value
        return None
