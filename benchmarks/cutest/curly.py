import jax.numpy as jnp
from jaxtyping import Array

from .problem import AbstractUnconstrainedMinimisation


class CURLY20(AbstractUnconstrainedMinimisation, strict=True):
    """The CURLY 20 function.

    A banded function with semi-bandwidth 20 and
    negative curvature near the starting point.

    Note J. Haffner --------------------------------------------------------------------
    The value q is created by the matrix-vector product of the mask and y. The mask has
    the form:

    [***  ]
    [ *** ]
    [  ***]
    [   **]
    [    *]

    And q = M @ y.
    ------------------------------------------------------------------------------------

    Source: Nick Gould, September 1997.

    Classification: OUR2-AN-V-0
    """

    n: int  # Number of dimensions. Options listed in SIF file: 100, 1000, 10000
    k: int  # Semi-bandwidth.
    mask: Array

    def __init__(self, n: int = 1000, k: int = 20):
        def create_mask(n, k):
            row_indices = jnp.arange(n)[:, None]
            col_indices = jnp.arange(n)[None, :]

            # A cell (i,j) should be 1 if i ≤ j ≤ min(i+k, n-1)
            min_indices = jnp.minimum(row_indices + k, n - 1)
            mask = (col_indices >= row_indices) & (col_indices <= min_indices)
            return mask

        self.n = n
        self.k = k
        self.mask = create_mask(n, k)

    def objective(self, y, args):
        del args
        q = self.mask @ y
        result = q * (q * (q**2 - 20) - 0.1)
        return jnp.sum(result)

    def y0(self):
        i = jnp.arange(1, self.n + 1)
        return 0.0001 * i / (self.n + 1)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# Create minimal subclasses for the different semi-bandwidths to conform with SIF names
class CURLY10(CURLY20):
    """The CURLY 10 function.

    A banded function with semi-bandwidth 10 and
    negative curvature near the starting point.

    Source: Nick Gould.

    SIF input: Nick Gould, September 1997.
    Classification: OUR2-AN-V-0
    """

    def __init__(self, n: int = 1000, k: int = 10):
        super().__init__(n=n, k=k)


class CURLY30(CURLY20):
    """The CURLY 30 function.

    A banded function with semi-bandwidth 30 and
    negative curvature near the starting point.

    Source: Nick Gould.
    SIF input: Nick Gould, September 1997.

    Classification: OUR2-AN-V-0
    """

    def __init__(self, n: int = 1000, k: int = 30):
        super().__init__(n=n, k=k)
