from ..custom_types import sentinel
from ..line_search import AbstractGLS, AbstractModel
from ..minimise import minimise
from .iterative_dual import IndirectIterativeDual
from .quasi_newton import AbstractQuasiNewton
from .trust_region import ClassicalTrustRegion


class GaussNewton(AbstractQuasiNewton):
    line_search: AbstractGLS = sentinel
    model: AbstractModel = sentinel

    def __post_init__(self):

        if self.line_search == sentinel:
            raise ValueError("No line search initialized in GaussNewton")

        if self.model == sentinel:
            raise ValueError("No model initialized in GaussNewton")

    def step(self, problem, y, args, options, state):
        if not self.model.gauss_newton:
            raise ValueError(
                "A model with gauss_newton=False was passed to \
                GeneralGaussNewton. Please use a different quasi-Newton method for \
                this."
            )
        line_search = self.line_search(self.model)
        sol = minimise(problem, line_search, y, args, options)
        new_y, new_state = self.update_solution(y, sol, state)

        return new_y, new_state, sol.state.aux


#
# Yep, LevenbergMarquardt is just an alias of GaussNewton with a default
# choice of model and line search. This is similar to the popular implementation
# of Moré (1978) "The Levenberg-Marquardt Algorithm: Implementation and Theory."
#


class LevenbergMarquardt(GaussNewton):
    line_search: AbstractGLS = sentinel
    model: AbstractModel = sentinel

    def __post_init__(self):
        if self.model == sentinel:
            self.model = IndirectIterativeDual(
                gauss_newton=True,
                atol=1e-6,
                rtol=1e-6,
                lambda_0=0.5,
            )
        if self.line_search == sentinel:
            self.line_search = ClassicalTrustRegion()

    def step(self, problem, y, args, options, state):
        super().step(problem, y, args, options, state)
