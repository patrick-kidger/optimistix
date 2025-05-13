from .arwhead import ARWHEAD as ARWHEAD
from .cosine import COSINE as COSINE
from .curly import CURLY10 as CURLY10, CURLY20 as CURLY20, CURLY30 as CURLY30
from .eg import EG2 as EG2
from .engval import ENGVAL1 as ENGVAL1, ENGVAL2 as ENGVAL2
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
    COSINE(n=10),
    COSINE(n=100),
    COSINE(n=1000),
    COSINE(n=10000),
    CURLY10(n=100),
    CURLY10(n=1000),
    CURLY10(n=10000),
    CURLY20(n=100),
    CURLY20(n=1000),
    CURLY20(n=10000),
    CURLY30(n=100),
    CURLY30(n=1000),
    CURLY30(n=10000),
    EG2(n=1000),
    ENGVAL1(n=2),
    ENGVAL1(n=50),
    ENGVAL1(n=100),
    ENGVAL1(n=1000),
    ENGVAL1(n=5000),
    ENGVAL2(),
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
