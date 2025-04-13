from .backtracking import BacktrackingArmijo as BacktrackingArmijo
from .best_so_far import (
    BestSoFarFixedPoint as BestSoFarFixedPoint,
    BestSoFarLeastSquares as BestSoFarLeastSquares,
    BestSoFarMinimiser as BestSoFarMinimiser,
    BestSoFarRootFinder as BestSoFarRootFinder,
)
from .bfgs import AbstractBFGS as AbstractBFGS, BFGS as BFGS, BFGS_B as BFGS_B
from .bisection import Bisection as Bisection
from .boundary_maps import (
    AbstractBoundaryMap as AbstractBoundaryMap,
    BoxProjection as BoxProjection,
    ClosestFeasiblePoint as ClosestFeasiblePoint,
)
from .cauchy_point import (
    cauchy_point as cauchy_point,
    CauchyNewton as CauchyNewton,
    CauchyNewtonDescent as CauchyNewtonDescent,
)
from .coleman_li import (
    ColemanLi as ColemanLi,
    ColemanLiDescent as ColemanLiDescent,
)
from .dogleg import Dogleg as Dogleg, DoglegDescent as DoglegDescent
from .filtered import FilteredLineSearch as FilteredLineSearch
from .fixed_point import FixedPointIteration as FixedPointIteration
from .gauss_newton import (
    AbstractGaussNewton as AbstractGaussNewton,
    GaussNewton as GaussNewton,
    NewtonDescent as NewtonDescent,
)
from .gradient_methods import (
    AbstractGradientDescent as AbstractGradientDescent,
    GradientDescent as GradientDescent,
    SteepestDescent as SteepestDescent,
)
from .interior_point import (
    InteriorDescent as InteriorDescent,
    InteriorPoint as InteriorPoint,
)
from .ipoptlike import (
    IPOPTLike as IPOPTLike,
    IPOPTLikeDescent as IPOPTLikeDescent,  # TODO: name might change
)
from .learning_rate import LearningRate as LearningRate
from .levenberg_marquardt import (
    DampedNewtonDescent as DampedNewtonDescent,
    IndirectDampedNewtonDescent as IndirectDampedNewtonDescent,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LevenbergMarquardt as LevenbergMarquardt,
)
from .nelder_mead import NelderMead as NelderMead
from .newton_chord import Chord as Chord, Newton as Newton
from .nonlinear_cg import (
    dai_yuan as dai_yuan,
    fletcher_reeves as fletcher_reeves,
    hestenes_stiefel as hestenes_stiefel,
    NonlinearCG as NonlinearCG,
    NonlinearCGDescent as NonlinearCGDescent,
    polak_ribiere as polak_ribiere,
)
from .optax import OptaxMinimiser as OptaxMinimiser
from .quasi_newton import (
    AbstractQuasiNewton as AbstractQuasiNewton,
    AbstractQuasiNewtonUpdate as AbstractQuasiNewtonUpdate,
    BFGS as BFGS,
    BFGSUpdate as BFGSUpdate,
    DFP as DFP,
    DFPUpdate as DFPUpdate,
from .proximal_projected import ProjectedGradientDescent as ProjectedGradientDescent
from .sequential import (
    QuadraticSubproblemDescent as QuadraticSubproblemDescent,
    SLSQP as SLSQP,
)
from .trust_region import (
    ClassicalTrustRegion as ClassicalTrustRegion,
    LinearTrustRegion as LinearTrustRegion,
)
