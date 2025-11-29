from contextlib import nullcontext as does_not_raise

import optimistix as optx
import pytest
from equinox import EquinoxTracetimeError


@pytest.mark.parametrize(
    "c1, c2, min_stepsize, expectation",
    [
        (
            0.1,
            0.2,
            1e-6,
            does_not_raise(),
        ),
        (
            0.2,
            0.1,
            1e-6,
            pytest.raises(EquinoxTracetimeError, match="between `c1` and 1"),
        ),
        (
            0.1,
            0.2,
            -1e3,
            pytest.raises(EquinoxTracetimeError, match="must be strictly greater"),
        ),
    ],
)
def test_zoom_wrong_params_in_post_init(c1, c2, min_stepsize, expectation):
    with expectation:
        optx.Zoom(c1=c1, c2=c2, min_stepsize=min_stepsize)
