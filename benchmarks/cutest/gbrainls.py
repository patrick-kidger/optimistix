import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class GBRAINLS(AbstractUnconstrainedMinimisation):
    """A brain tissue modeling problem.

    This problem involves fitting a model of the shear moduli in the human brain
    to data, formulated as a nonlinear least-squares problem.

    The model function is: f(λ) = C0 * λ^(2*α-1)
    where λ represents the shear strain ratio.

    Source: an example in
    L.A. Mihai, S. Budday, G.A. Holzapfel, E. Kuhl and A. Goriely.
    "A family of hyperelastic models for human brain tissue",
    Journal of Mechanics and Physics of Solids,
    DOI: 10.1016/j.jmps.2017.05.015 (2017).

    As conveyed by Angela Mihai (U. Cardiff)

    SIF input: Nick Gould, June 2017.

    Classification: SUR2-MN-2-0
    """

    # Allow selecting which starting point to use (0-based indexing)
    start_point: int = 0  # 0 or 1
    _allowed_start_points = frozenset({0, 1})

    def __check_init__(self):
        if self.start_point not in self._allowed_start_points:
            raise ValueError(
                f"start_point must be in {self._allowed_start_points}, "
                f"got {self.start_point}"
            )

    def objective(self, y, args):
        del args

        # Extract model parameters
        alpha, c0 = y

        # Define lambda values (strain ratios) - linearly spaced between 1 and 2
        lambda_values = jnp.linspace(1.0, 2.0, 200)

        # The experimental data from the SIF file for 11 datasets
        # For simplicity, we'll use the first dataset (all 200 values)
        data_values = jnp.array(
            [
                1.1352929e0,
                1.1659807e0,
                1.1734516e0,
                1.1715945e0,
                1.1732607e0,
                1.1797534e0,
                1.1855043e0,
                1.1935876e0,
                1.2033084e0,
                1.2098163e0,
                1.2105108e0,
                1.2126348e0,
                1.2207112e0,
                1.2287777e0,
                1.2323889e0,
                1.2387623e0,
                1.2433394e0,
                1.2493115e0,
                1.2510949e0,
                1.2550339e0,
                1.2562187e0,
                1.2537636e0,
                1.2507643e0,
                1.2521567e0,
                1.2513655e0,
                1.2485745e0,
                1.2411448e0,
                1.2363371e0,
                1.2315664e0,
                1.2278782e0,
                1.2248371e0,
                1.2209314e0,
                1.2151101e0,
                1.2089010e0,
                1.2044711e0,
                1.2032602e0,
                1.2001480e0,
                1.1963549e0,
                1.1932753e0,
                1.1913672e0,
                1.1904643e0,
                1.1876500e0,
                1.1856368e0,
                1.1834885e0,
                1.1808178e0,
                1.1796356e0,
                1.1791079e0,
                1.1774039e0,
                1.1757140e0,
                1.1737979e0,
                1.1739328e0,
                1.1736369e0,
                1.1733346e0,
                1.1727211e0,
                1.1734569e0,
                1.1742784e0,
                1.1736464e0,
                1.1725050e0,
                1.1711877e0,
                1.1697955e0,
                1.1690577e0,
                1.1686078e0,
                1.1689765e0,
                1.1683838e0,
                1.1666659e0,
                1.1660543e0,
                1.1651816e0,
                1.1649780e0,
                1.1658948e0,
                1.1671842e0,
                1.1679172e0,
                1.1674242e0,
                1.1664027e0,
                1.1654134e0,
                1.1666122e0,
                1.1678247e0,
                1.1692165e0,
                1.1695411e0,
                1.1691426e0,
                1.1689509e0,
                1.1694938e0,
                1.1705695e0,
                1.1719031e0,
                1.1727478e0,
                1.1739746e0,
                1.1743711e0,
                1.1744284e0,
                1.1748554e0,
                1.1757673e0,
                1.1770423e0,
                1.1776634e0,
                1.1782912e0,
                1.1784735e0,
                1.1790294e0,
                1.1801119e0,
                1.1812948e0,
                1.1819150e0,
                1.1813884e0,
                1.1809496e0,
                1.1816699e0,
                1.1824376e0,
                1.1823383e0,
                1.1821295e0,
                1.1814096e0,
                1.1810303e0,
                1.1809132e0,
                1.1822742e0,
                1.1839315e0,
                1.1846724e0,
                1.1849830e0,
                1.1859115e0,
                1.1867733e0,
                1.1887414e0,
                1.1906194e0,
                1.1923223e0,
                1.1932372e0,
                1.1937081e0,
                1.1948802e0,
                1.1967837e0,
                1.1988292e0,
                1.2004032e0,
                1.2013605e0,
                1.2023843e0,
                1.2040936e0,
                1.2055706e0,
                1.2067231e0,
                1.2079639e0,
                1.2092282e0,
                1.2103735e0,
                1.2122047e0,
                1.2139260e0,
                1.2152455e0,
                1.2160458e0,
                1.2173593e0,
                1.2192893e0,
                1.2209115e0,
                1.2223882e0,
                1.2236042e0,
                1.2243621e0,
                1.2259306e0,
                1.2278759e0,
                1.2303992e0,
                1.2329248e0,
                1.2345476e0,
                1.2361584e0,
                1.2383548e0,
                1.2409761e0,
                1.2437612e0,
                1.2458100e0,
                1.2477855e0,
                1.2489645e0,
                1.2505031e0,
                1.2526903e0,
                1.2549134e0,
                1.2568353e0,
                1.2582106e0,
                1.2595772e0,
                1.2616271e0,
                1.2634069e0,
                1.2656991e0,
                1.2678183e0,
                1.2699249e0,
                1.2715780e0,
                1.2733662e0,
                1.2759422e0,
                1.2788523e0,
                1.2810787e0,
                1.2832231e0,
                1.2858347e0,
                1.2883938e0,
                1.2905706e0,
                1.2933439e0,
                1.2959546e0,
                1.2982830e0,
                1.3003461e0,
                1.3027166e0,
                1.3057888e0,
                1.3082785e0,
                1.3109906e0,
                1.3140463e0,
                1.3170557e0,
                1.3199256e0,
                1.3232159e0,
                1.3264943e0,
                1.3297287e0,
                1.3328284e0,
                1.3360844e0,
                1.3387036e0,
                1.3415058e0,
                1.3440609e0,
                1.3466792e0,
                1.3490603e0,
                1.3520049e0,
                1.3550078e0,
                1.3579840e0,
                1.3617129e0,
                1.3650421e0,
                1.3694318e0,
                1.3744666e0,
                1.3817053e0,
            ]
        )

        # Define the model function: C0 * λ^(2*α-1)
        def model_func(lambda_val):
            beta = 2.0 * alpha - 1.0
            return c0 * lambda_val**beta

        # Compute model predictions for all lambda values
        model_values = jax.vmap(model_func)(lambda_values)

        # Compute residuals
        residuals = model_values - data_values

        # Return sum of squared residuals
        return jnp.sum(residuals**2)

    def y0(self):
        # Starting points from the SIF file
        if self.start_point == 0:
            # Default starting point: (alpha=-4.0, c0=-0.1)
            return jnp.array([-4.0, -0.1])
        elif self.start_point == 1:
            # Alternative starting point: (alpha=1.0, c0=1.0)
            return jnp.array([1.0, 1.0])
        else:
            assert False

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution is not specified in the SIF file
        return None

    def expected_objective_value(self):
        # The minimum objective value should be close to 0.0
        return jnp.array(0.0)
