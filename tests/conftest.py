import pytest

from .helpers import getkey as _getkey


@pytest.fixture
def getkey():
    return _getkey
