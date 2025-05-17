from .arwhead import ARWHEAD as ARWHEAD
from .cosine import COSINE as COSINE
from .curly import CURLY10 as CURLY10, CURLY20 as CURLY20, CURLY30 as CURLY30
from .denschn import (
    DENSCHNA as DENSCHNA,
    DENSCHNB as DENSCHNB,
    DENSCHNC as DENSCHNC,
    DENSCHND as DENSCHND,
    DENSCHNE as DENSCHNE,
    DENSCHNF as DENSCHNF,
)
from .devgla import DEVGLA1 as DEVGLA1, DEVGLA2 as DEVGLA2
from .dixmaa import (
    DIXMAANA1 as DIXMAANA1,
    DIXMAANB as DIXMAANB,
    DIXMAANC as DIXMAANC,
    DIXMAAND as DIXMAAND,
    DIXMAANE1 as DIXMAANE1,
    DIXMAANF as DIXMAANF,
    DIXMAANG as DIXMAANG,
    DIXMAANH as DIXMAANH,
)
from .dixmaai import (
    DIXMAANI1 as DIXMAANI1,
    DIXMAANJ as DIXMAANJ,
    DIXMAANK as DIXMAANK,
    DIXMAANL as DIXMAANL,
)
from .dixmaam import (
    DIXMAANM1 as DIXMAANM1,
    DIXMAANN as DIXMAANN,
    DIXMAANO as DIXMAANO,
    DIXMAANP as DIXMAANP,
)
from .dmn import (
    DMN15102LS as DMN15102LS,
    DMN15103LS as DMN15103LS,
    DMN15332LS as DMN15332LS,
    DMN15333LS as DMN15333LS,
    DMN37142LS as DMN37142LS,
    DMN37143LS as DMN37143LS,
)
from .eg import EG2 as EG2
from .elatvidu import ELATVIDU as ELATVIDU
from .engval import ENGVAL1 as ENGVAL1, ENGVAL2 as ENGVAL2
from .exp_scipy import EXP2 as EXP2, EXP2B as EXP2B, EXP2NE as EXP2NE
from .fletch import FLETCBV3 as FLETCBV3
from .fminsurf import FMINSRF2 as FMINSRF2, FMINSURF as FMINSURF
from .freuroth import FREURONE as FREURONE, FREUROTH as FREUROTH
from .gaussian import GAUSSIAN as GAUSSIAN
from .gulf import GULF as GULF
from .hahn1ls import HAHN1LS as HAHN1LS
from .hairy import HAIRY as HAIRY
from .hatfld import (
    HATFLDD as HATFLDD,
    HATFLDE as HATFLDE,
    HATFLDFL as HATFLDFL,
    HATFLDFLS as HATFLDFLS,
    HATFLDGLS as HATFLDGLS,
)
from .heart import HEART6LS as HEART6LS, HEART8LS as HEART8LS
from .helix import HELIX as HELIX
from .hilbert import HILBERTA as HILBERTA, HILBERTB as HILBERTB
from .himmelblau import (
    HIMMELBB as HIMMELBB,
    HIMMELBCLS as HIMMELBCLS,
    HIMMELBF as HIMMELBF,
    HIMMELBG as HIMMELBG,
    HIMMELBH as HIMMELBH,
)
from .humps import HUMPS as HUMPS
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
    DENSCHNA(),
    DENSCHNB(),
    DENSCHNC(),
    DENSCHND(),
    DENSCHNE(),
    DENSCHNF(),
    DEVGLA1(),
    DEVGLA2(),
    DIXMAANA1(n=100),
    DIXMAANA1(n=500),
    DIXMAANA1(n=1000),
    DIXMAANB(n=100),
    DIXMAANB(n=500),
    DIXMAANB(n=1000),
    DIXMAANC(n=100),
    DIXMAANC(n=500),
    DIXMAANC(n=1000),
    DIXMAAND(n=100),
    DIXMAAND(n=500),
    DIXMAAND(n=1000),
    DIXMAANE1(n=300),
    DIXMAANE1(n=1500),
    DIXMAANE1(n=3000),
    DIXMAANF(n=300),
    DIXMAANF(n=1500),
    DIXMAANF(n=3000),
    DIXMAANG(n=300),
    DIXMAANG(n=1500),
    DIXMAANG(n=3000),
    DIXMAANH(n=300),
    DIXMAANH(n=1500),
    DIXMAANH(n=3000),
    DIXMAANI1(n=300),
    DIXMAANI1(n=1500),
    DIXMAANI1(n=3000),
    DIXMAANJ(n=300),
    DIXMAANJ(n=1500),
    DIXMAANJ(n=3000),
    DIXMAANK(n=300),
    DIXMAANK(n=1500),
    DIXMAANK(n=3000),
    DIXMAANL(n=300),
    DIXMAANL(n=1500),
    DIXMAANL(n=3000),
    DIXMAANM1(n=300),
    DIXMAANM1(n=1500),
    DIXMAANM1(n=3000),
    DIXMAANN(n=300),
    DIXMAANN(n=1500),
    DIXMAANN(n=3000),
    DIXMAANO(n=300),
    DIXMAANO(n=1500),
    DIXMAANO(n=3000),
    DIXMAANP(n=300),
    DIXMAANP(n=1500),
    DIXMAANP(n=3000),
    EG2(n=1000),
    ENGVAL1(n=2),
    ENGVAL1(n=50),
    ENGVAL1(n=100),
    ENGVAL1(n=1000),
    ENGVAL1(n=5000),
    ENGVAL2(),
    EXP2(),
    EXP2B(),
    EXP2NE(),
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
    #    FMINSURF(),  # TODO: has a bug
    #    FMINSRF2(),  # TODO: has a bug
    FREUROTH(n=2),
    FREUROTH(n=10),
    FREUROTH(n=50),
    FREUROTH(n=100),
    FREUROTH(n=500),
    FREUROTH(n=1000),
    FREUROTH(n=5000),
    FREURONE(n=2),
    FREURONE(n=10),
    FREURONE(n=50),
    FREURONE(n=100),
    FREURONE(n=500),
    FREURONE(n=1000),
    FREURONE(n=5000),
    GAUSSIAN(),
    GULF(),
    HAHN1LS(),
    HAHN1LS(y0_id=1),
    HAIRY(),
    HATFLDD(),
    HATFLDE(),
    HATFLDFL(),
    HATFLDFLS(),
    HATFLDGLS(),
    HEART6LS(),
    HEART8LS(),
    HELIX(),
    HILBERTA(n=2),
    HILBERTA(n=4),
    HILBERTA(n=5),
    HILBERTA(n=6),
    HILBERTA(n=10),
    HILBERTB(n=5),
    HILBERTB(n=10),
    HILBERTB(n=50),
    HIMMELBB(),
    HIMMELBCLS(),
    HIMMELBF(),
    HIMMELBG(),
    HIMMELBH(),
    HUMPS(),
    ROSENBR(),
)
