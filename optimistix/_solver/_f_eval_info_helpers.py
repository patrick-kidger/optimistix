from collections.abc import Callable

from jaxtyping import Scalar

from .._custom_types import Y
from .._misc import lin_to_grad
from .._search import FunctionInfo


def make_eval_info_for_search(
    info_needed: type[FunctionInfo],
    y_eval: Y,
    f_eval: Scalar,
    lin_fn: Callable[[Y], Scalar],
    autodiff_mode: str,
) -> FunctionInfo.Eval | FunctionInfo.EvalGrad:
    """Build the cheapest scalar `FunctionInfo` satisfying `info_needed`."""
    if issubclass(FunctionInfo.Eval, info_needed):
        return FunctionInfo.Eval(f_eval)
    elif issubclass(FunctionInfo.EvalGrad, info_needed):
        grad = lin_to_grad(lin_fn, y_eval, autodiff_mode, f_eval.dtype)
        return FunctionInfo.EvalGrad(f_eval, grad)
    else:
        raise ValueError(
            f"Cannot provide requested function information at `y_eval`: {info_needed}."
            " Only FunctionInfo.Eval and FunctionInfo.EvalGrad are supported."
        )


def promote_to_eval_grad(
    f_eval_info: FunctionInfo.Eval | FunctionInfo.EvalGrad,
    y_eval: Y,
    f_eval: Scalar,
    lin_fn: Callable[[Y], Scalar],
    autodiff_mode: str,
) -> FunctionInfo.EvalGrad:
    """Reuse an existing scalar gradient evaluation, or compute one if needed."""
    if isinstance(f_eval_info, FunctionInfo.EvalGrad):
        return f_eval_info
    else:
        grad = lin_to_grad(lin_fn, y_eval, autodiff_mode, f_eval.dtype)
        return FunctionInfo.EvalGrad(f_eval, grad)
