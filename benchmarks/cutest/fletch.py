import jax.numpy as jnp

from .problem import AbstractUnconstrainedMinimisation


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FLETCHBV(AbstractUnconstrainedMinimisation, strict=True):
    """The FLETCHBV function.

    Another Boundary Value problem.

    Source: The first problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    N.B. This formulation is incorrect. See FLETCBV2.SIF for
    the correct version.

    SIF input: Nick Gould, Oct 1992.
    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # Default dimension in SIF file
    kappa: float = 1.0  # Parameter used in the problem

    def objective(self, y, args):
        del args
        h = 1.0 / (self.n + 1)
        h2 = h * h

        # Define components as described in the SIF file
        # First term: 0.5 * (x_1 + x_2)^2
        f1 = 0.5 * (y[0] + y[1]) ** 2

        # Terms G(i) for i=1...n-1: 0.5 * (x_i - x_{i+1})^2
        f2 = 0.5 * jnp.sum((y[:-1] - y[1:]) ** 2)

        # Terms L(i): involving -2/h^2 * x_i
        f3 = -2.0 / h2 * jnp.sum(y[:-1]) - (1.0 + 2.0 / h2) * y[-1]

        # Terms C(i): involving -kappa/h^2 * cos(x_i)
        f4 = -self.kappa / h2 * jnp.sum(jnp.cos(y))

        return f1 + f2 + f3 + f4

    def y0(self):
        # Initial values from SIF file: i*h for i=1..n
        h = 1.0 / (self.n + 1)
        return jnp.arange(1, self.n + 1) * h

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FLETBV3M(AbstractUnconstrainedMinimisation, strict=True):
    """The FLETBV3M function.

    Variant of FLETCBV3, another boundary value problem, by Luksan et al

    Source: problem 30 in
    L. Luksan, C. Matonoha and J. Vlcek
    Modified CUTE problems for sparse unconstrained optimization
    Technical Report 1081
    Institute of Computer Science
    Academy of Science of the Czech Republic

    based on a scaled version of the first problem given by
    R. Fletcher,
    "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    SIF input: Nick Gould, June, 2013
    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # Default dimension in SIF file
    kappa: float = 1.0  # Parameter used in the problem
    objscale: float = 1.0e8  # Scaling factor

    def objective(self, y, args):
        del args
        h = 1.0 / (self.n + 1)
        h2 = h * h
        p = 1.0 / self.objscale

        # Define each term based on the SIF file
        # G(0): p * 0.5 * (x_1)^2
        f1 = 0.5 * p * (y[0]) ** 2

        # G(i) for i=1...n-1: p * 0.5 * (x_i - x_{i+1})^2
        f2 = 0.5 * p * jnp.sum((y[:-1] - y[1:]) ** 2)

        # G(n): p * 0.5 * (x_n)^2
        f3 = 0.5 * p * (y[-1]) ** 2

        # C(i): p * cos(x_i) * (-kappa/h^2)
        f4 = p * (-self.kappa / h2) * jnp.sum(jnp.cos(y))

        # S(i): 100 * sin(0.01 * x_i) * p * (-1-2/h^2)
        f5 = p * (-1.0 - 2.0 / h2) * jnp.sum(100.0 * jnp.sin(0.01 * y))

        return f1 + f2 + f3 + f4 + f5

    def y0(self):
        # Initial values from SIF file: i*h for i=1..n
        h = 1.0 / (self.n + 1)
        return jnp.arange(1, self.n + 1) * h

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FLETCBV2(AbstractUnconstrainedMinimisation, strict=True):
    """The FLETCBV2 function.

    Another Boundary Value problem.

    Source: The first problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.
    but assuming that the 1/h**2 term should read h**2
    This is what Fletcher intended (private communication).

    The author comments: "The problem arises from discretizing the bvp
                 x"=-2+sin x in [0,1]
    with x(0)=0, x(1)=1. This gives a symmetric system of equations,
    the residual vector of which is the gradient of the given function."
    He multiplies through by h^2 before integrating.

    SIF input: Nick Gould, Nov 1992.
    Classification: OUR2-AN-V-0
    """

    n: int = 5000  # Default dimension in SIF file
    kappa: float = 1.0  # Parameter used in the problem

    def objective(self, y, args):
        del args
        h = 1.0 / (self.n + 1)
        h2 = h * h

        # Define components as described in the SIF file
        # G(0): 0.5 * (x_1)^2
        f1 = 0.5 * (y[0]) ** 2

        # G(i) for i=1...n-1: 0.5 * (x_i - x_{i+1})^2
        f2 = 0.5 * jnp.sum((y[:-1] - y[1:]) ** 2)

        # G(n): 0.5 * (x_n)^2
        f3 = 0.5 * (y[-1]) ** 2

        # L(i) for i=1...n-1: x_i * (-2*h2)
        f4 = -2.0 * h2 * jnp.sum(y[:-1])

        # L(n): x_n * (-1-2*h2)
        f5 = (-1.0 - 2.0 * h2) * y[-1]

        # C(i): -kappa*h2 * cos(x_i)
        f6 = -self.kappa * h2 * jnp.sum(jnp.cos(y))

        return f1 + f2 + f3 + f4 + f5 + f6

    def y0(self):
        # Initial values from SIF file: i*h for i=1..n
        h = 1.0 / (self.n + 1)
        return jnp.arange(1, self.n + 1) * h

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: This implementation requires human review and verification against
# another CUTEst interface
class FLETCHCR(AbstractUnconstrainedMinimisation, strict=True):
    """The FLETCHCR function.

    The chained Rosenbrock function as given by Fletcher.

    Source: The second problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    SIF input: Nick Gould, Oct 1992.
    Classification: OUR2-AN-V-0
    """

    n: int = 1000  # Default dimension in SIF file

    def objective(self, y, args):
        del args

        # Based on the SIF file, this is a chained Rosenbrock function
        # Efficient vectorized computation
        term1 = jnp.sum((y[1:] - 0.01 * y[:-1] ** 2) ** 2)
        term2 = jnp.sum((y[:-1] - 1.0) ** 2)  # This is the SQ2 term from the SIF

        return term1 + term2

    def y0(self):
        # Starting point: all zeros
        return jnp.zeros(self.n)

    def args(self):
        return None

    def expected_result(self):
        # Minimum value is 0, achieved when all variables are 1
        return jnp.ones(self.n)

    def expected_objective_value(self):
        return jnp.array(0.0)


# TODO: this has not yet been compared against another interface to CUTEst
class FLETCBV3(AbstractUnconstrainedMinimisation, strict=True):
    """The FLETCBV3 function.

    A boundary value problem from Fletcher (1992).

    Source: The first problem given by
    R. Fletcher, "An optimal positive definite update for sparse Hessian matrices"
    Numerical Analysis report NA/145, University of Dundee, 1992.

    Note J. Haffner --------------------------------------------------------------------
    The reference given appears to be incorrect, the PDF available under the title above
    does not include a problem description.

    This can be defined for different dimensions (original SIF allows 10, 100, 1000,
    5000, or 10000), with 5000 being the default in the SIF file.
    ------------------------------------------------------------------------------------

    SIF input: Nick Gould, Oct 1992.
    Classification: OUR2-AN-V-0
    """

    n: int = 5000
    scale: float = 1e8  # Called OBJSCALE in the SIF file
    extra_term: int = 1  # Corresponds to the parameter kappa, which is 1 or 0

    def objective(self, y, args):
        p, c = args
        h = 1.0 / (self.n + 1)
        h2 = h * h

        f1 = (y[0] + y[1]) ** 2
        f2 = jnp.sum((y[:-1] - y[1:]) ** 2)
        f3 = (h2 + 2) / h2 * jnp.sum(y)
        f4 = c / h2 * jnp.sum(jnp.cos(y))

        return 0.5 * p * (f1 + f2) - p * f3 + f4

    def y0(self):
        n = self.n
        h = 1.0 / (self.n + 1)
        # Starting point according to SIF file: i*h for i=1..n
        return jnp.arange(1, n + 1) * h

    def args(self):
        # p and kappa from SIF file
        p = 1 / self.scale
        c = self.extra_term
        return jnp.array([p, c])

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None  # Takes different values for different problem configurations
