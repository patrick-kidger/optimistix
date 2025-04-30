from .cutest import (
    AbstractConstrainedMinimisation,
    AbstractUnconstrainedMinimisation,
    problems,
)


unconstrained_problems = (
    p for p in problems if isinstance(p(), AbstractUnconstrainedMinimisation)
)

constrained_problems = (
    p for p in problems if isinstance(p(), AbstractConstrainedMinimisation)
)
