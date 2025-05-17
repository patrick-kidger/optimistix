import jax
import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class CYCLOOCFLS(AbstractUnconstrainedMinimisation):
    """The cyclooctane molecule configuration problem (least-squares version).

    The cyclooctane molecule is comprised of eight carbon atoms aligned
    in an equally spaced ring. When they take a position of minimum
    potential energy so that next-neighbours are equally spaced.

    Given positions v_1, ..., v_p in R^3 (with p = 8 for cyclooctane),
    and given a spacing c^2 we have that

       ||v_i - v_i+1,mod p||^2 = c^2 for i = 1,..,p, and
       ||v_i - v_i+2,mod p||^2 = 2p/(p-2) c^2

    where (arbitrarily) we have v_1 = 0 and component 1 of v_2 = 0

    Source:
    an extension of the cyclooctane molecule configuration space as
    described in (for example)

     E. Coutsias, S. Martin, A. Thompson & J. Watson
     "Topology of cyclooctane energy landscape"
     J. Chem. Phys. 132-234115 (2010)

    SIF input: Nick Gould, Feb 2020.

    This is a least-squares version of CYCLOOCF.SIF

    Classification: SUR2-MN-V-0
    """

    p: int = 8  # Number of molecules (default 8, can be 100, 1000, 10000, 100000)
    n: int = 0  # Number of variables (will be set in __init__)

    def __init__(self):
        # The number of variables is determined by the number of molecules
        # We have 3 coordinates for each molecule except:
        # - v1 is fixed at (0,0,0), so we don't need variables for it
        # - x2 is fixed at 0, so we only need y2 and z2
        # Total: 3*p - 4
        self.n = 3 * self.p - 4

    def objective(self, y, args):
        del args
        p = self.p
        c = 1.0  # Radius parameter from SIF file

        # Pre-compute some constants
        c2 = c * c
        sc2 = (2.0 * p / (p - 2.0)) * c2

        # Reshape the input vector into the proper coordinates
        # First organize the variables into a comprehensible structure
        # v1 is fixed at (0,0,0)
        # v2 has x2=0, and y2, z2 are the first two variables
        v = jnp.zeros((p, 3))
        # Set v2 (v2_x is 0)
        v = v.at[1, 1].set(y[0])  # v2_y = y2
        v = v.at[1, 2].set(y[1])  # v2_z = z2

        # Set v3 to vp
        idx = 2
        for i in range(2, p):
            v = v.at[i, 0].set(y[idx])  # vi_x
            v = v.at[i, 1].set(y[idx + 1])  # vi_y
            v = v.at[i, 2].set(y[idx + 2])  # vi_z
            idx += 3

        # Compute squared distances for adjacent molecules (A groups)
        # ||v_i - v_i+1,mod p||^2 = c^2 for i = 1,..,p
        a_residuals = jnp.zeros(p)

        def compute_a_residual(i):
            i_next = (i + 1) % p
            squared_dist = jnp.sum((v[i] - v[i_next]) ** 2)
            return squared_dist - c2

        a_residuals = jax.vmap(compute_a_residual)(jnp.arange(p))

        # Compute squared distances for next-next molecules (B groups)
        # ||v_i - v_i+2,mod p||^2 = 2p/(p-2) c^2 for i = 1,..,p
        b_residuals = jnp.zeros(p)

        def compute_b_residual(i):
            i_next_next = (i + 2) % p
            squared_dist = jnp.sum((v[i] - v[i_next_next]) ** 2)
            return squared_dist - sc2

        b_residuals = jax.vmap(compute_b_residual)(jnp.arange(p))

        # Combine all residuals and compute sum of squares
        all_residuals = jnp.concatenate([a_residuals, b_residuals])
        return jnp.sum(all_residuals**2)

    def y0(self):
        # Initial values from SIF file
        # Simple approach: distribute molecules evenly on a ring in the xy-plane
        p = self.p
        radius = jnp.sqrt(1.0)  # c = 1.0
        angles = jnp.arange(p) * (2 * jnp.pi / p)

        # Compute coordinates for each molecule
        v = jnp.zeros((p, 3))
        v = v.at[:, 0].set(radius * jnp.cos(angles))  # x
        v = v.at[:, 1].set(radius * jnp.sin(angles))  # y
        # z remains 0

        # Convert to the expected variable format
        y = jnp.zeros(self.n)
        y = y.at[0].set(v[1, 1])  # y2
        y = y.at[1].set(v[1, 2])  # z2

        idx = 2
        for i in range(2, p):
            y = y.at[idx].set(v[i, 0])  # vi_x
            y = y.at[idx + 1].set(v[i, 1])  # vi_y
            y = y.at[idx + 2].set(v[i, 2])  # vi_z
            idx += 3

        return y

    def args(self):
        return None

    def expected_result(self):
        # The SIF file suggests there are multiple solutions (different conformations)
        return None

    def expected_objective_value(self):
        # According to the SIF file, the minimal objective value is 0.0
        return jnp.array(0.0)
