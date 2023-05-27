from .backtracking import BacktrackingArmijo
from .bfgs import BFGS
from .bisection import Bisection
from .descent import (
    NormalisedGradient,
    NormalisedNewton,
    UnnormalisedGradient,
    UnnormalisedNewton,
)
from .dogleg import Dogleg
from .fixed_point import FixedPointIteration
from .iterative_dual import DirectIterativeDual, IndirectIterativeDual
from .levenberg_marquardt_gauss_newton import (
    AbstractGaussNewton,
    GaussNewton,
    IndirectLevenbergMarquardt,
    LevenbergMarquardt,
)
from .nelder_mead import NelderMead
from .newton_chord import Chord, Newton
from .nonlinear_cg import GradOnly, NonlinearCG
from .nonlinear_cg_descent import (
    dai_yuan,
    fletcher_reeves,
    hestenes_stiefel,
    NonlinearCGDescent,
    polak_ribiere,
)
from .trust_region import ClassicalTrustRegion
