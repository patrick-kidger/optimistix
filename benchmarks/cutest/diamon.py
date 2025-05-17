from .problem import AbstractUnconstrainedMinimisation


# TODO: This is a placeholder implementation that requires proper implementation
# The DIAMON problems are complex powder diffraction data fitting problems
# with a large number of variables and data points.
class DIAMON2DLS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction data fitting problem (2D version).

    This problem involves fitting a model to powder diffraction data from
    Diamond Light Source. It has 66 variables and 4643 data points.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version: Nick Gould, Jan 2020, corrected May 2024

    Classification: SUR2-MN-66-0
    """

    n: int = 66  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def y0(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def args(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def expected_result(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def expected_objective_value(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )


# TODO: This is a placeholder implementation that requires proper implementation
# The DIAMON problems are complex powder diffraction data fitting problems
# with a large number of variables and data points.
class DIAMON3DLS(AbstractUnconstrainedMinimisation):
    """Diamond powder diffraction data fitting problem (3D version).

    This problem involves fitting a model to powder diffraction data from
    Diamond Light Source. It has 99 variables and 4643 data points.

    Source: Data from Aaron Parsons, I14: Hard X-ray Nanoprobe,
    Diamond Light Source, Harwell, Oxfordshire, England, EU.

    SIF input: Nick Gould and Tyrone Rees, Feb 2016
    Least-squares version: Nick Gould, Jan 2020, corrected May 2024

    Classification: SUR2-MN-99-0
    """

    n: int = 99  # Number of variables

    def objective(self, y, args):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def y0(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def args(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def expected_result(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )

    def expected_objective_value(self):
        raise NotImplementedError(
            "This complex problem requires further implementation"
        )
