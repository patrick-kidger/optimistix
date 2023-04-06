from ..least_squares import AbstractLeastSquaresSolver


class DirectLevenbergMarquardt(AbstractLeastSquaresSolver):
    def init(self, problem, y, args, options):
        ...

    def step(self, problem, y, args, options, state):
        ...

    def terminate(self, problem, y, args, options, state):
        ...
