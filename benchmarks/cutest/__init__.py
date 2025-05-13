from .arwhead import ARWHEAD as ARWHEAD
from .curly import CURLY10 as CURLY10, CURLY20 as CURLY20, CURLY30 as CURLY30
from .fletch import FLETCBV3 as FLETCBV3
from .himmelblau import HIMMELBG as HIMMELBG
from .problem import (
    AbstractUnconstrainedMinimisation as AbstractUnconstrainedMinimisation,
)
from .rosenbr import ROSENBR as ROSENBR


problems = (
    ARWHEAD(n=100),
    ARWHEAD(n=500),
    ARWHEAD(n=1000),
    ARWHEAD(n=5000),
    CURLY10(n=100),
    CURLY10(n=1000),
    CURLY10(n=10000),
    CURLY20(n=100),
    CURLY20(n=1000),
    CURLY20(n=10000),
    CURLY30(n=100),
    CURLY30(n=1000),
    CURLY30(n=10000),
    # Not varying the scale term in the FLETCBV3 problem
    FLETCBV3(n=10, extra_term=1),
    FLETCBV3(n=100, extra_term=1),
    FLETCBV3(n=1000, extra_term=1),
    FLETCBV3(n=5000, extra_term=1),
    FLETCBV3(n=10000, extra_term=1),
    FLETCBV3(n=10, extra_term=0),
    FLETCBV3(n=100, extra_term=0),
    FLETCBV3(n=1000, extra_term=0),
    FLETCBV3(n=5000, extra_term=0),
    FLETCBV3(n=10000, extra_term=0),
    HIMMELBG(),
    ROSENBR(),
)
