from .problem import AbstractUnconstrainedMinimisation


# TODO: proper implementation required
class HYDC20LS(AbstractUnconstrainedMinimisation, strict=True):
    """The hydrocarbon-20 problem by Fletcher (least-squares version).

    Source: Problem 2b in
    J.J. More', "A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer Seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: N. Gould and Ph. Toint, Feb 1991.

    Classification: SUR2-AN-99-00
    """

    def objective(self, y, args):
        raise NotImplementedError("HYDC20LS implementation not completed")

    def y0(self):
        raise NotImplementedError("HYDC20LS implementation not completed")

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None


# TODO: proper implementation required
class HYDCAR6LS(AbstractUnconstrainedMinimisation, strict=True):
    """The hydrocarbon-6 problem by Fletcher (least-squares version).

    Source: Problem 2a in
    J.J. More', "A collection of nonlinear model problems"
    Proceedings of the AMS-SIAM Summer Seminar on the Computational
    Solution of Nonlinear Systems of Equations, Colorado, 1988.
    Argonne National Laboratory MCS-P60-0289, 1989.

    SIF input: N. Gould and Ph. Toint, Feb 1991.

    Classification: SUR2-AN-29-00
    """

    def objective(self, y, args):
        raise NotImplementedError("HYDCAR6LS implementation not completed")

    def y0(self):
        raise NotImplementedError("HYDCAR6LS implementation not completed")

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
