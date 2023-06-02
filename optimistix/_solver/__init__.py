from .backtracking import BacktrackingArmijo as BacktrackingArmijo
from .bfgs import BFGS as BFGS
from .bisection import Bisection as Bisection
from .descent import NormalisedGradient as NormalisedGradient
from .descent import NormalisedNewton as NormalisedNewton
from .descent import UnnormalisedGradient as UnnormalisedGradient
from .descent import UnnormalisedNewton as UnnormalisedNewton
from .dogleg import Dogleg as Dogleg
from .fixed_point import FixedPointIteration as FixedPointIteration
from .iterative_dual import DirectIterativeDual as DirectIterativeDual
from .iterative_dual import IndirectIterativeDual as IndirectIterativeDual
from .levenberg_marquardt_gauss_newton import AbstractGaussNewton as AbstractGaussNewton
from .levenberg_marquardt_gauss_newton import GaussNewton as GaussNewton
from .levenberg_marquardt_gauss_newton import (
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
)
from .levenberg_marquardt_gauss_newton import LevenbergMarquardt as LevenbergMarquardt
from .newton_chord import Chord as Chord
from .newton_chord import Newton as Newton
from .nonlinear_cg import GradOnly as GradOnly
from .nonlinear_cg import NonlinearCG as NonlinearCG
from .nonlinear_cg_descent import dai_yuan as dai_yuan
from .nonlinear_cg_descent import fletcher_reeves as fletcher_reeves
from .nonlinear_cg_descent import hestenes_stiefel as hestenes_stiefel
from .nonlinear_cg_descent import NonlinearCGDescent as NonlinearCGDescent
from .nonlinear_cg_descent import polak_ribiere as polak_ribiere
from .trust_region import ClassicalTrustRegion as ClassicalTrustRegion
