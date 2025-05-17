import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: Human review
# TODO: This implementation requires verification against another CUTEst interface
class CHAINWOO(AbstractUnconstrainedMinimisation):
    """The chained Woods problem, a variant on Woods function.

    This problem is a sum of n/2 sets of 6 terms, each of which is
    assigned its own group. For a given set i, the groups are
    A(i), B(i), C(i), D(i), E(i) and F(i). Groups A(i) and C(i) contain 1
    nonlinear element each, denoted Y(i) and Z(i).

    This version uses a slightly unorthodox expression of Woods
    function as a sum of squares (see Buckley).

    Source: problem 8 in
    A.R.Conn,N.I.M.Gould and Ph.L.Toint,
    "Testing a class of methods for solving minimization
    problems with simple bounds on their variables,
    Mathematics of Computation 50, pp 399-430, 1988.

    SIF input: Nick Gould and Ph. Toint, Dec 1995.

    Classification: SUR2-AN-V-0
    """

    n: int = 4000  # Dimension of the problem (2*ns + 2)
    ns: int = 1999  # Number of sets (default 1999, which gives n=4000)

    def objective(self, y, args):
        del args

        # Extract the number of sets
        ns = self.ns

        # Define function to compute terms for a single set i
        def compute_set_terms(i):
            # Compute indices for this set
            # j starts at 4 and increases by 2 for each set
            j = 4 + 2 * i

            # Indices for the variables in this set
            j_3 = j - 3
            j_2 = j - 2
            j_1 = j - 1

            # Group A(i): 0.01 * (x_{j-2} - x_{j-3}^2)^2
            a_i = 0.01 * (y[j_2 - 1] - y[j_3 - 1] ** 2) ** 2

            # Group B(i): -1.0 - x_{j-3}
            b_i = (-1.0 - y[j_3 - 1]) ** 2

            # Group C(i): (1/90) * (x_j - x_{j-1}^2)^2
            c_i = (1.0 / 90.0) * (y[j - 1] - y[j_1 - 1] ** 2) ** 2

            # Group D(i): -1.0 - x_{j-1}
            d_i = (-1.0 - y[j_1 - 1]) ** 2

            # Group E(i): 0.1 * (x_{j-2} + x_j - 2.0)^2
            e_i = 0.1 * (y[j_2 - 1] + y[j - 1] - 2.0) ** 2

            # Group F(i): 10.0 * (x_{j-2} - x_j)^2
            f_i = 10.0 * (y[j_2 - 1] - y[j - 1]) ** 2

            # Return all terms for this set
            return jnp.array([a_i, b_i, c_i, d_i, e_i, f_i])

        # Create array of set indices
        set_indices = jnp.arange(ns)

        # Compute terms for all sets using vmap
        all_terms = jax.vmap(compute_set_terms)(set_indices)

        # Flatten and sum all terms
        return jnp.sum(all_terms)

    def y0(self):
        # Initial values from SIF file
        y_init = jnp.full(self.n, -2.0)

        # Override specific elements
        y_init = y_init.at[0].set(-3.0)  # X1
        y_init = y_init.at[1].set(-1.0)  # X2
        y_init = y_init.at[2].set(-3.0)  # X3
        y_init = y_init.at[3].set(-1.0)  # X4

        return y_init

    def args(self):
        return None

    def expected_result(self):
        # The optimal solution has all components equal to 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        # SIF file comment (line 154): optimal objective value is 0.0
        return jnp.array(0.0)
