from .backtracking import BacktrackingArmijo
from .bfgs import BFGS
from .bisection import Bisection
from .cg import CG
from .cholesky import Cholesky
from .descent import (
    NormalisedGradient,
    NormalisedNewton,
    NormalisedNewtonInverse,
    UnnormalisedGradient,
    UnnormalisedNewton,
    UnnormalisedNewtonInverse,
)
from .diagonal import Diagonal
from .dogleg import Dogleg
from .fixed_point import FixedPointIteration
from .iterative_dual import DirectIterativeDual, IndirectIterativeDual
from .levenberg_marquardt_gauss_newton import (
    AbstractGaussNewton,
    GaussNewton,
    IndirectLevenbergMarquardt,
    LevenbergMarquardt,
)
from .lu import LU
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
from .qr import QR
from .svd import SVD
from .triangular import Triangular
from .tridiagonal import Tridiagonal
from .trust_region import ClassicalTrustRegion
