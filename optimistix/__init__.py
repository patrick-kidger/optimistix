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

from . import internal as internal
from ._adjoint import (
    AbstractAdjoint as AbstractAdjoint,
    ImplicitAdjoint as ImplicitAdjoint,
    RecursiveCheckpointAdjoint as RecursiveCheckpointAdjoint,
)
from ._descent import AbstractDescent as AbstractDescent
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
from ._minimise import (
    AbstractMinimiser as AbstractMinimiser,
    minimise as minimise,
)
from ._misc import (
    max_norm as max_norm,
    rms_norm as rms_norm,
    two_norm as two_norm,
)
from ._root_find import (
    AbstractRootFinder as AbstractRootFinder,
    root_find as root_find,
)
from ._solution import RESULTS as RESULTS, Solution as Solution
from ._solver import (
    AbstractBFGS as AbstractBFGS,
    AbstractGaussNewton as AbstractGaussNewton,
    AbstractGradientDescent as AbstractGradientDescent,
    AbstractNonlinearCG as AbstractNonlinearCG,
    AbstractTrustRegion as AbstractTrustRegion,
    BacktrackingArmijo as BacktrackingArmijo,
    BFGS as BFGS,
    Bisection as Bisection,
    Chord as Chord,
    ClassicalTrustRegion as ClassicalTrustRegion,
    dai_yuan as dai_yuan,
    DirectIterativeDual as DirectIterativeDual,
    Dogleg as Dogleg,
    DoglegDescent as DoglegDescent,
    FixedPointIteration as FixedPointIteration,
    fletcher_reeves as fletcher_reeves,
    GaussNewton as GaussNewton,
    Gradient as Gradient,
    GradientDescent as GradientDescent,
    hestenes_stiefel as hestenes_stiefel,
    IndirectIterativeDual as IndirectIterativeDual,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LearningRate as LearningRate,
    LevenbergMarquardt as LevenbergMarquardt,
    LinearTrustRegion as LinearTrustRegion,
    NelderMead as NelderMead,
    Newton as Newton,
    NewtonDescent as NewtonDescent,
    NonlinearCG as NonlinearCG,
    NonlinearCGDescent as NonlinearCGDescent,
    OptaxMinimiser as OptaxMinimiser,
    polak_ribiere as polak_ribiere,
)


__version__ = "0.0.1"
