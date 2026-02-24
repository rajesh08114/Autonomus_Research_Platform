from __future__ import annotations

from src.core.subprocess_runner import classify_error


def test_classify_error():
    assert classify_error("ModuleNotFoundError: No module named x") == "ModuleNotFoundError"

