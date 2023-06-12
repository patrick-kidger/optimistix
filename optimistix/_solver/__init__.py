# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from .learning_rate import LearningRate as LearningRate
from .levenberg_marquardt_gauss_newton import (
    AbstractGaussNewton as AbstractGaussNewton,
    GaussNewton as GaussNewton,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LevenbergMarquardt as LevenbergMarquardt,
)
from .nelder_mead import NelderMead as NelderMead
from .newton_chord import Chord as Chord, Newton as Newton
from .nonlinear_cg import (
    GradientDescent as GradientDescent,
    GradOnly as GradOnly,
    NonlinearCG as NonlinearCG,
)
from .nonlinear_cg_descent import (
    dai_yuan as dai_yuan,
    fletcher_reeves as fletcher_reeves,
    hestenes_stiefel as hestenes_stiefel,
    NonlinearCGDescent as NonlinearCGDescent,
    polak_ribiere as polak_ribiere,
)
from .trust_region import ClassicalTrustRegion as ClassicalTrustRegion
