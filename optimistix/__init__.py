from . import _internal as _internal
from ._adjoint import (
    AbstractAdjoint as AbstractAdjoint,
    ImplicitAdjoint as ImplicitAdjoint,
    RecursiveCheckpointAdjoint as RecursiveCheckpointAdjoint,
)
from ._fixed_point import (
    AbstractFixedPointSolver as AbstractFixedPointSolver,
    fixed_point as fixed_point,
)
from ._iterate import (
    AbstractIterativeSolver as AbstractIterativeSolver,
    iterative_solve as iterative_solve,
)
from ._least_squares import (
    AbstractLeastSquaresSolver as AbstractLeastSquaresSolver,
    least_squares as least_squares,
)
from ._line_search import AbstractDescent as AbstractDescent
from ._minimise import (
    AbstractMinimiser as AbstractMinimiser,
    minimise as minimise,
)
from ._root_find import (
    AbstractRootFinder as AbstractRootFinder,
    root_find as root_find,
)
from ._solution import RESULTS as RESULTS, Solution as Solution
from ._solver import (
    AbstractGaussNewton as AbstractGaussNewton,
    BacktrackingArmijo as BacktrackingArmijo,
    BFGS as BFGS,
    Bisection as Bisection,
    Chord as Chord,
    ClassicalTrustRegion as ClassicalTrustRegion,
    dai_yuan as dai_yuan,
    DirectIterativeDual as DirectIterativeDual,
    Dogleg as Dogleg,
    FixedPointIteration as FixedPointIteration,
    fletcher_reeves as fletcher_reeves,
    GaussNewton as GaussNewton,
    GradientDescent as GradientDescent,
    GradOnly as GradOnly,
    hestenes_stiefel as hestenes_stiefel,
    IndirectIterativeDual as IndirectIterativeDual,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LearningRate as LearningRate,
    LevenbergMarquardt as LevenbergMarquardt,
    Newton as Newton,
    NonlinearCG as NonlinearCG,
    NonlinearCGDescent as NonlinearCGDescent,
    NormalisedGradient as NormalisedGradient,
    NormalisedNewton as NormalisedNewton,
    polak_ribiere as polak_ribiere,
    UnnormalisedGradient as UnnormalisedGradient,
    UnnormalisedNewton as UnnormalisedNewton,
)


__version__ = "0.0.1"
