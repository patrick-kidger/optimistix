from .cutest import AbstractUnconstrainedMinimisation, problems


cutest_unconstrained_problems = (
    p for p in problems if isinstance(p, AbstractUnconstrainedMinimisation)
)
