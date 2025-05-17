from .problem import AbstractUnconstrainedMinimisation


# TODO: This problem needs human implementation due to its complexity
class JIMACK(AbstractUnconstrainedMinimisation, strict=True):
    """A complex 3D finite element problem from nonlinear elasticity.

    Source: Parallel optimization for a finite element problem
    from nonlinear elasticity, P. K. Jimack, School of Computer Studies,
    U. of Leeds, report 91.22, July 1991.

    SIF input: Nick Gould, September 1991.
    Classification: OUR2-AN-3549-0

    This problem requires a 3-D discretization of the box [-1,1]x[-1,1]x[0,DELTA]
    with a complex finite element formulation that's challenging to implement in JAX.
    Human implementation is needed for accuracy.
    """

    def objective(self, y, args):
        """Compute the objective function for the JIMACK problem.

        This involves a complex 3D finite element calculation with many variables.
        """
        raise NotImplementedError("JIMACK problem requires human implementation")

    def y0(self):
        """Initial point for the problem (identity map in the original formulation)."""
        raise NotImplementedError("JIMACK problem requires human implementation")

    def args(self):
        """Additional arguments for the objective function."""
        raise NotImplementedError("JIMACK problem requires human implementation")

    def expected_result(self):
        """Expected result of the optimization problem."""
        raise NotImplementedError("JIMACK problem requires human implementation")

    def expected_objective_value(self):
        """Expected value of the objective function at the optimal solution."""
        raise NotImplementedError("JIMACK problem requires human implementation")
