"""A feasible incumbent must not be reported as a verified MIP optimum."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import mip
from src.model import Unit


def test_time_limited_incumbent_is_rejected(monkeypatch):
    unit = Unit("test", 0, 1, 10, 0, 1, 1, 1, 1, 1, 1, 1)
    result = SimpleNamespace(success=False, x=np.zeros(5), fun=0.0,
                             message="Time limit reached")
    monkeypatch.setattr(mip, "milp", lambda *args, **kwargs: result)
    with pytest.raises(RuntimeError, match="Time limit reached"):
        mip.solve_deterministic_mip(unit, [20.0], 0, 1)
