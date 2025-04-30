import equinox as eqx
import jax.numpy as jnp


class HS35(eqx.Module):
    """The HS35 problem from the CUTEST collection of benchmark problems.

    Source: Problem 35 in W. Hock and K. Schittkowski,
    "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187,
    Springer Verlag, Heidelberg, 1981.

    SIF input: A.R. Conn, April 1990
    Classification: QLR2-AN-3-1
    """

    def objective(self, y, args):
        del args
        # Classic QP of the form 0.5 x^T G x + c^T x + d
        G = jnp.array([[4.0, 2.0, 2.0], [2.0, 4.0, 0.0], [2.0, 0.0, 2.0]])
        c = jnp.array([-8.0, -6.0, -4.0])
        d = jnp.array([-9.0])

        value = 0.5 * jnp.dot(y.T, jnp.matmul(G, y)) + jnp.dot(c.T, y) + d
        return jnp.squeeze(value)

    def y0(self):
        return 0.5 * jnp.array([1.0, 1.0, 1.0])

    def args(self):
        return None

    def constraint(self, y):
        # Linear constraints of the form Ax + b >= 0 (with A diagonal)
        a = jnp.array([-1.0, -1.0, -2.0])
        b = jnp.array([-3.0])
        inequality_constraint = jnp.dot(a, y) + b
        return None, jnp.squeeze(inequality_constraint)

    def bounds(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return jnp.array(0.1111111111)
