from .backtracking import AbstractBacktrackingGLS, BacktrackingArmijo
from .iterative_dual import DirectIterativeDual, IndirectIterativeDual
from .models import (
    NormalizedGradient,
    NormalizedNewton,
    UnnormalizedGradient,
    UnnormalizedNewton,
)
from .trust_region import ClassicalTrustRegion
