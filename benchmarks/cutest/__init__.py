from .bt import BT1 as BT1, BT2 as BT2, BT4 as BT4, BT5 as BT5
from .flt import FLT as FLT
from .hatfldh import HATFLDH as HATFLDH
from .problem import (
    AbstractBoundedMinimisation as AbstractBoundedMinimisation,
    AbstractConstrainedMinimisation as AbstractConstrainedMinimisation,
    AbstractUnconstrainedMinimisation as AbstractUnconstrainedMinimisation,
)
from .rosenbr import ROSENBR as ROSENBR
from .snake import SNAKE as SNAKE


problems = (
    BT1(),
    BT2(),
    *(BT4(i) for i in BT4().provided_y0s),
    *(BT5(i) for i in BT5().provided_y0s),
    FLT(),
    HATFLDH(),
    ROSENBR(),
    SNAKE(),
)
