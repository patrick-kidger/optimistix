from ..line_search import AbstractGLS, AbstractModel
from ..minimise import minimise
from ..search import ClassicalTrustRegion, DirectIterativeDual, IndirectIterativeDual
from .quasi_newton import AbstractQuasiNewton


class LevenbergMarquardt(AbstractQuasiNewton):
    def step(self, problem, y, args, options, state):
        model = DirectIterativeDual(gauss_newton=True)
        line_search = ClassicalTrustRegion(model)
        sol = minimise(problem, line_search, y, args, options)
        new_y, new_state = self.update_solution(y, sol, state)

        return new_y, new_state, sol.state.aux


class LevenbergMarquardtIndirect(AbstractQuasiNewton):
    def step(self, problem, y, args, options, state):
        model = IndirectIterativeDual(
            gauss_newton=True, atol=1e-5, rtol=1e-4, lambda_0=1
        )
        line_search = ClassicalTrustRegion(model)
        sol = minimise(problem, line_search, y, args, options)
        new_y, new_state = self.update_solution(y, sol, state)

        return new_y, new_state, sol.state.aux


class GeneralGaussNewton(AbstractQuasiNewton):
    line_search: AbstractGLS
    model: AbstractModel

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
