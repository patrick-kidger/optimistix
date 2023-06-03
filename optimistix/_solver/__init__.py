from .backtracking import BacktrackingArmijo as BacktrackingArmijo
from .bfgs import BFGS as BFGS
from .bisection import Bisection as Bisection
from .descent import (
    NormalisedGradient as NormalisedGradient,
    NormalisedNewton as NormalisedNewton,
    UnnormalisedGradient as UnnormalisedGradient,
    UnnormalisedNewton as UnnormalisedNewton,
)
from .dogleg import Dogleg as Dogleg
from .fixed_point import FixedPointIteration as FixedPointIteration
from .iterative_dual import (
    DirectIterativeDual as DirectIterativeDual,
    IndirectIterativeDual as IndirectIterativeDual,
)
from .levenberg_marquardt_gauss_newton import (
    AbstractGaussNewton as AbstractGaussNewton,
    GaussNewton as GaussNewton,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LevenbergMarquardt as LevenbergMarquardt,
)
from .newton_chord import Chord as Chord, Newton as Newton
from .nonlinear_cg import GradOnly as GradOnly, NonlinearCG as NonlinearCG
from .nonlinear_cg_descent import (
    dai_yuan as dai_yuan,
    fletcher_reeves as fletcher_reeves,
    hestenes_stiefel as hestenes_stiefel,
    NonlinearCGDescent as NonlinearCGDescent,
    polak_ribiere as polak_ribiere,
)
from .trust_region import ClassicalTrustRegion as ClassicalTrustRegion
