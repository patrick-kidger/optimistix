from .cutest import (
    AbstractConstrainedMinimisation,
    AbstractUnconstrainedMinimisation,
    problems,
)


cutest_unconstrained_problems = (
    p for p in problems if isinstance(p, AbstractUnconstrainedMinimisation)
)

cutest_constrained_problems = (
    p for p in problems if isinstance(p, AbstractConstrainedMinimisation)
)
