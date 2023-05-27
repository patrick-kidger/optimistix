from . import internal
from .fixed_point import AbstractFixedPointSolver, fixed_point, FixedPointProblem
from .iterate import AbstractIterativeProblem, AbstractIterativeSolver, iterative_solve
from .least_squares import (
    AbstractLeastSquaresSolver,
    least_squares,
    LeastSquaresProblem,
)
from .line_search import AbstractDescent
from .minimise import AbstractMinimiser, minimise, MinimiseProblem
from .root_find import AbstractRootFinder, root_find, RootFindProblem
from .solution import RESULTS, Solution
from .solver import (
    AbstractGaussNewton,
    BacktrackingArmijo,
    BFGS,
    Bisection,
    Chord,
    ClassicalTrustRegion,
    dai_yuan,
    DirectIterativeDual,
    Dogleg,
    FixedPointIteration,
    fletcher_reeves,
    GaussNewton,
    GradOnly,
    hestenes_stiefel,
    IndirectIterativeDual,
    IndirectLevenbergMarquardt,
    LevenbergMarquardt,
    NelderMead,
    Newton,
    NonlinearCG,
    NonlinearCGDescent,
    NormalisedGradient,
    NormalisedNewton,
    polak_ribiere,
    UnnormalisedGradient,
    UnnormalisedNewton,
)


__version__ = "0.0.1"
