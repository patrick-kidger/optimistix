"""
TODO: This implementation requires human review and may need substantial rework.

This file implements a simplified approximation of the HIELOW problem from CUTEst.
The actual implementation requires access to a FORTRAN subroutine named "HIELOW" 
which is called in the SIF file but not included in it. The SIF file contains:

```fortran
DOUBLE PRECISION FUNCTION F(B1,B2,B3)
...
CALL HIELOW(.FALSE.,.FALSE.)
```

The HIELOW problem models a hierarchical logit function for transportation choice:
- Car
- Public transportation (with Theta0/THE1 parameter)
  - Bus
  - Tram

The coefficients represent:
- BET1, BET2: Elemental coefficients affecting utility functions 
  (related to time and cost)
- THE1: A structural coefficient for the hierarchical model's nesting structure

Without the actual FORTRAN implementation, this file provides only an approximate
implementation based on general hierarchical logit model principles. The results
may not match the original CUTEst problem.
"""

import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: human review required - this is a simplified approximation
# as the original FORTRAN implementation is not available
class HIELOW(AbstractUnconstrainedMinimisation, strict=True):
    """Hierarchical logit model for modal choice prediction.

    This problem involves finding the parameters of a hierarchical logit
    model that maximize the likelihood of a sample of observations.

    Source:
    Example provided with HieLoW: modal choices with randomly drawn data.
    HieLoW: Hierarchical Logit for Windows, written by Michel Bierlaire

    SIF input: automatically produced by HieLoW

    Classification: OUR2-MN-3-0

    Description of the model:
    - Hierarchical structure with car and public transportation options
    - Public transportation includes bus and tram
    - Three parameters: two elemental coefficients (BET1, BET2)
      and one structural coefficient (THE1)
    """

    def objective(self, y, args):
        del args

        # Extract individual variables
        # BET1, BET2: elemental coefficients
        # THE1: structural coefficient
        bet1, bet2, the1 = y

        # The actual objective function in HIELOW.SIF is defined through
        # a complex sequence of FORTRAN code related to hierarchical logit
        # modeling. Since we do not have access to this specific function,
        # we implement a simplified version that captures its key characteristics.
        #
        # Based on logit modeling literature, a typical likelihood function:
        # - penalizes parameter values far from reasonable ranges
        # - has a negative log-likelihood form
        # - involves exponential utilities weighted by structural parameters

        # Simplified negative log-likelihood function based on common patterns
        # in hierarchical logit models

        # Compute utility functions (simplified versions)
        u_bus = bet1 * 1.5 + bet2 * 2.0
        u_tram = bet1 * 1.0 + bet2 * 2.5

        # Apply structural parameter
        u_pt = jnp.log(jnp.exp(u_bus / the1) + jnp.exp(u_tram / the1)) * the1
        u_car = bet1 * 0.8 + bet2 * 3.0

        # Compute probabilities
        p_car = jnp.exp(u_car) / (jnp.exp(u_car) + jnp.exp(u_pt))
        p_pt = jnp.exp(u_pt) / (jnp.exp(u_car) + jnp.exp(u_pt))
        p_bus = (
            jnp.exp(u_bus / the1)
            / (jnp.exp(u_bus / the1) + jnp.exp(u_tram / the1))
            * p_pt
        )
        p_tram = (
            jnp.exp(u_tram / the1)
            / (jnp.exp(u_bus / the1) + jnp.exp(u_tram / the1))
            * p_pt
        )

        # Regularization term to constrain THE1 to be positive
        reg_the1 = jnp.exp(-10.0 * jnp.minimum(the1, 0.0))

        # For a simplified objective (without actual data), we use a heuristic measure
        # that makes sense for logit models: we aim to have reasonable probability
        # distributions and parameter values
        neg_log_likelihood = (
            -10.0
            * (
                jnp.log(p_car + 1e-10)
                + jnp.log(p_bus + 1e-10)
                + jnp.log(p_tram + 1e-10)
            )
            + 0.1 * (bet1**2 + bet2**2 + (the1 - 1.0) ** 2)  # Regularization
            + reg_the1
        )

        return neg_log_likelihood

    def y0(self):
        # Initial values from SIF file
        # BET1=0.0, BET2=0.0, THE1=1.0
        return jnp.array([0.0, 0.0, 1.0])

    def args(self):
        return None

    def expected_result(self):
        # The exact solution is not clearly specified in the SIF file
        return None

    def expected_objective_value(self):
        # From SIF file, there's a value of 874.16543206, but it's unclear
        # if this is the optimal value or just a reference
        # Given our simplified implementation, this value won't be meaningful
        return None
