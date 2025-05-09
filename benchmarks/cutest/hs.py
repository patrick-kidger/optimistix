import jax.numpy as jnp

from .problem import AbstractConstrainedMinimisation


# TODO(viljoen): flagging this problem as interesting for validation of regularisation
# and refinement methods (JH)
class HS13(AbstractConstrainedMinimisation):
    """The HS13 problem from the CUTEST collection of benchmark problems.

    A problem in 2 variables where constraint qualification does not hold.

    Source: Problem 13 in W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187,
    Springer Verlag, Heidelberg, 1981.

    Note J. Haffner: -------------------------------------------------------------------
    Original reference: https://doi.org/10.1007/978-3-642-48320-2_4

    This problem has a simple parabolic objective function, but the constraint function
    is cubic in x1. The optimal value coincides with the x-intercept of the constraint
    function, which is also an inflection point for the constraint function.
    This creates a feasible region with a pointy tip that gets very narrow close to the
    solution.
    ------------------------------------------------------------------------------------

    SIF input: A.R. Conn March 1990
    Classification: QOR2-AN-2-1
    """

    def objective(self, y, args):
        del args
        x1, x2 = y
        return (x1 - 2.0) ** 2 + x2**2

    def y0(self):
        return jnp.array([-2.0, -2.0])  # This is infeasible (J. Haffner)

    def args(self):
        return None

    def constraint(self, y):
        x1, x2 = y
        c1 = (1.0 - x1) ** 3 - x2
        return None, c1

    def bounds(self):
        # Compared to CUTEST.jl, and original reference: nonnegativity constraints on y.
        # This is not clear from the SIF file alone. (J. Haffner)
        return jnp.zeros_like(self.y0()), jnp.full_like(self.y0(), jnp.inf)

    def expected_result(self):
        return jnp.array([1.0, 0.0])  # From original reference (J. Haffner)

    def expected_objective_value(self):
        return jnp.array(1.0)
