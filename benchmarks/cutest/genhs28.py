import equinox as eqx
import jax.numpy as jnp


# TODO: this problem needs to be verified against another CUTEST interface
class GENHS28(eqx.Module):
    """The GENHS28 problem from the CUTEST collection of benchmark problems.

    Source: A multi-dimensional extension of problem 28 in
    W. Hock and K. Schittkowski, "Test examples for nonlinear programming codes",
    Lectures Notes in Economics and Mathematical Systems 187,
    Springer Verlag, Heidelberg, 1981.
    Available at: https://doi.org/10.1007/978-3-642-48320-2 (TODO what page?)

    SIF input: Nick Gould, December 1991, minor correction by Ph. Shott, Jan 1995.
    Classification: QLR2-AY-10-8
    """

    def objective(self, y, args):
        del args
        # TODO: make evaluation more performant
        # From GROUP USES and ELEMENT TYPE sections:
        # OBJ is the sum of SQ(X(i), X(i+1)) for i = 1 to N-1
        # Where SQ(V1, V2) = (V1 + V2)^2
        obj_value = 0.0
        for i in range(10 - 1):
            u1 = y[i] + y[i + 1]  # U1 = V1 + V2
            obj_value += u1**2
        return obj_value

    def y0(self):
        # From START POINT section:
        # X1 = -4.0, rest = 1.0
        return jnp.ones(10).at[0].set(-4.0)

    def args(self):
        return None

    def constraint(self, y):
        # TODO: make evaluation more performant
        # From GROUPS section:
        # CON(i) = X(i) + 2*X(i+1) + 3*X(i+2) - 1.0 for i = 1 to N-2
        constraints = []
        for i in range(10 - 2):
            con_i = y[i] + 2.0 * y[i + 1] + 3.0 * y[i + 2] - 1.0
            constraints.append(con_i)

        return jnp.array(constraints), None

    def bounds(self):
        # From BOUNDS section: all variables are free (FR)
        return None

    def expected_result(self):
        # The optimal point coordinates are not provided
        return None

    def expected_objective_value(self):
        # From OBJECT BOUND section, LO SOLTN/GENHS28 value
        return 0.0
