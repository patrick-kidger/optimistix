from equinox.internal import (
    AbstractProgressMeter as EqxAbstractProgressMeter,
    NoProgressMeter as EqxNoProgressMeter,
    TextProgressMeter as EqxTextProgressMeter,
    TqdmProgressMeter as EqxTqdmProgressMeter,
)


def _inherit_doc(cls):
    [base] = [base for base in cls.__bases__ if base.__module__ != __name__]
    cls.__doc__ = base.__doc__
    return cls


@_inherit_doc
class AbstractProgressMeter(EqxAbstractProgressMeter):
    pass


@_inherit_doc
class NoProgressMeter(EqxNoProgressMeter, AbstractProgressMeter):
    pass


@_inherit_doc
class TextProgressMeter(EqxTextProgressMeter, AbstractProgressMeter):
    pass


@_inherit_doc
class TqdmProgressMeter(EqxTqdmProgressMeter, AbstractProgressMeter):
    pass
