from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "scripts"
    / "evaluate_voc2007.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_voc2007", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_percentile_handles_distribution_boundaries() -> None:
    values = [5, 1, 4, 2, 3]

    assert evaluator.percentile(values, 0.0) == 1
    assert evaluator.percentile(values, 0.5) == 3
    assert evaluator.percentile(values, 1.0) == 5


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_percentile_rejects_fraction_outside_unit_interval(fraction) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluator.percentile([1, 2, 3], fraction)
