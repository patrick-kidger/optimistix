from .cutest import AbstractUnconstrainedMinimisation, problems


unconstrained_problems = (
    p for p in problems if isinstance(p, AbstractUnconstrainedMinimisation)
)
