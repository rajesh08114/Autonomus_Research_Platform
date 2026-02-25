from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config.settings import settings


VSCODE_EXTENSION_MODE = "vscode_extension"
BACKEND_MODE = "backend"


def normalize_execution_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {VSCODE_EXTENSION_MODE, BACKEND_MODE}:
        return raw
    fallback = str(settings.EXECUTION_MODE or "").strip().lower()
    if fallback in {VSCODE_EXTENSION_MODE, BACKEND_MODE}:
        return fallback
    return VSCODE_EXTENSION_MODE


def is_vscode_execution_mode(state: Mapping[str, Any] | None) -> bool:
    mode = normalize_execution_mode((state or {}).get("execution_mode"))
    return mode == VSCODE_EXTENSION_MODE


def local_python_command() -> str:
    command = str(settings.LOCAL_PYTHON_COMMAND or "python").strip()
    return command or "python"
