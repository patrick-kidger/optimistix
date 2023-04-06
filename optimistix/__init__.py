from . import internal
from .fixed_point import AbstractFixedPointSolver, fixed_point, FixedPointProblem
from .least_squares import (
    AbstractLeastSquaresSolver,
    least_squares,
    LeastSquaresProblem,
)
from .line_search import AbstractGLS, AbstractModel
from .linear_operator import (
    AbstractLinearOperator,
    AddLinearOperator,
    AuxLinearOperator,
    ComposedLinearOperator,
    diagonal,
    DiagonalLinearOperator,
    DivLinearOperator,
    FunctionLinearOperator,
    has_unit_diagonal,
    IdentityLinearOperator,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_upper_triangular,
    JacobianLinearOperator,
    linearise,
    materialise,
    MatrixLinearOperator,
    MulLinearOperator,
    PyTreeLinearOperator,
    TaggedLinearOperator,
    TangentLinearOperator,
)
from .linear_solve import AbstractLinearSolver, AutoLinearSolver, linear_solve
from .linear_tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    transpose_tags,
    transpose_tags_rules,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)
from .minimise import AbstractMinimiser, minimise, MinimiseProblem
from .root_find import AbstractRootFinder, root_find, RootFindProblem
from .search import (
    AbstractBacktrackingGLS,
    NormalizedGradient,
    NormalizedNewton,
    UnnormalizedGradient,
    UnnormalizedNewton,
)
from .solution import RESULTS, Solution
from .solver import (
    Bisection,
    CG,
    Cholesky,
    Chord,
    Diagonal,
    FixedPointIteration,
    GaussNewton,
    LevenbergMarquardt,
    LU,
    Newton,
    QR,
    SVD,
    Triangular,
    Tridiagonal,
)


__version__ = "0.0.1"
