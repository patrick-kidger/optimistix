import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class DANIWOODLS(AbstractUnconstrainedMinimisation):
    """NIST Data fitting problem DANWOOD.

    This is a revised version of the original inaccurate formulation of DANWOODLS,
    with corrections provided by Abel Siqueira, Federal University of Parana.

    The model is: y = b1 * x^b2 + e

    Source: Problem from the NIST nonlinear regression test set
      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

    Daniel, C. and F. S. Wood (1980).
      Fitting Equations to Data, Second Edition.
      New York, NY: John Wiley and Sons, pp. 428-431.

    SIF input: Nick Gould and Tyrone Rees, Oct 2015 (as DANWOODLS)
              correction by Abel Siqueira, Feb 2019 (renamed DANIWOODLS)

    Classification: SUR2-MN-2-0
    """

    n: int = 2  # Number of variables
    m: int = 6  # Number of data points

    def objective(self, y, args):
        del args
        b1, b2 = y

        # Data points from the SIF file
        x_data = jnp.array([1.309, 1.471, 1.490, 1.565, 1.611, 1.680])
        y_data = jnp.array([2.138, 3.421, 3.597, 4.340, 4.882, 5.660])

        # Define the model: y = b1 * x^b2
        def model(x):
            return b1 * x**b2

        # Compute model predictions for all x values using vmap
        y_pred = jax.vmap(model)(x_data)

        # Compute residuals and return sum of squares
        residuals = y_pred - y_data
        return jnp.sum(residuals**2)

    def y0(self):
        # Initial values from SIF file (START1)
        return jnp.array([1.0, 5.0])

    def args(self):
        return None

    def expected_result(self):
        # According to NIST, the certified values are:
        # b1 = 0.76280 ± 0.13845
        # b2 = 4.0294 ± 0.89523
        return jnp.array([0.7628, 4.0294])

    def expected_objective_value(self):
        # Using the certified values, the residual sum of squares is
        # approximately 0.0040132
        return jnp.array(0.0040132)
