from .bt import BT1 as BT1, BT2 as BT2
from .flt import FLT as FLT
from .problem import (
    AbstractBoundedMinimisation as AbstractBoundedMinimisation,
    AbstractConstrainedMinimisation as AbstractConstrainedMinimisation,
    AbstractUnconstrainedMinimisation as AbstractUnconstrainedMinimisation,
)
from .rosenbr import ROSENBR as ROSENBR


problems = (ROSENBR(), FLT(), BT1(), BT2())
