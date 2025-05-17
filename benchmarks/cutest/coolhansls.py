import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class COOLHANSLS(AbstractUnconstrainedMinimisation):
    """A problem arising from the analysis of a Cooley-Hansen economy with
    loglinear approximation.

    The problem is to solve the matrix equation
               A * X * X + B * X + C = 0
    where A, B and C are known N times N matrices and X an unknown matrix
    of matching dimension. The instance considered here has N = 3.

    Source:
    S. Ceria, private communication, 1995.

    SIF input: Ph. Toint, Feb 1995.
    Least-squares version of COOLHANS.SIF, Nick Gould, Jan 2020.

    Classification: SUR2-RN-9-0
    """

    n: int = 9  # Number of variables (3x3 matrix X flattened)

    def objective(self, y, args):
        del args

        # Reshape the flattened variables into a 3x3 matrix
        X = y.reshape(3, 3)

        # Define matrices A, B, and C from the SIF file
        A = jnp.array([[0.0, 0.0, 0.0], [0.13725e-6, 937.62, -42.207], [0.0, 0.0, 0.0]])

        B = jnp.array(
            [
                [0.0060893, -44.292, 2.0011],
                [0.13880e-6, -1886.0, 42.362],
                [-0.13877e-6, 42.362, -2.0705],
            ]
        )

        C = jnp.array([[0.0, 44.792, 0.0], [0.0, 948.21, 0.0], [0.0, -42.684, 0.0]])

        # Compute A * X * X
        AXX = A @ X @ X

        # Compute B * X
        BX = B @ X

        # Matrix equation: A * X * X + B * X + C = 0
        # Residual matrix: A * X * X + B * X + C
        residual_matrix = AXX + BX + C

        # Compute the sum of squares of all elements in the residual matrix
        return jnp.sum(residual_matrix**2)

    def y0(self):
        # Initial values not explicitly specified in the SIF file,
        # but lines 95-98 suggest X(1,:) and X(3,:) might be fixed at 0
        # We'll use a small random initialization
        return 0.01 * jnp.ones(self.n)

    def args(self):
        return None

    def expected_result(self):
        # The SIF file doesn't specify the optimal solution
        # But line 146 suggests the objective value at the solution is 0.0,
        # meaning that a perfect solution to the matrix equation exists
        return None

    def expected_objective_value(self):
        # According to the SIF file, line 146
        return jnp.array(0.0)
