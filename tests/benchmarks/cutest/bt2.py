import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ScalarLike


class BT2(eqx.Module):
    """The BT2 problem from the CUTEST collection of benchmark problems.
    Original source: https://www.jstor.org/stable/2157674?seq=21, page 620. Similar in
    spirit to BT1 - the objective and constraint function have similar shapes, and form
    a shifted juxtaposition.
    Three starting values are documented in the original paper - a vector of all ones,
    all tens and all hundreds. In the official CUTEST collection, this value defaults to
    all tens, so we match that here.
    """

    multiply_y0: ScalarLike = 10.0

    def objective(self, y, args):
        x1, x2, x3 = y
        return (x1 - 1) ** 2 + (x1 - x2) ** 2 + (x2 - x3) ** 4

    def y0(self):
        return self.multiply_y0 * jnp.ones(3)

    def args(self):
        return None

    def constraint(self, y):
        x1, x2, x3 = y
        return x1 * (1 + x2**2) + x3**4 - 4 - 3 * jnp.sqrt(2), None

    def bounds(self):
        return None

    def expected_result(self):
        return jnp.array([1.1049, 1.1967, 1.5353])

    # TODO: add expected objective value: 0.032568200
