from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

from src.config.settings import settings


VSCODE_EXTENSION_MODE = "vscode_extension"
BACKEND_MODE = "backend"


def normalize_execution_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {VSCODE_EXTENSION_MODE, BACKEND_MODE}:
        return raw
    configured = str(settings.EXECUTION_MODE or "").strip().lower()
    if configured in {VSCODE_EXTENSION_MODE, BACKEND_MODE}:
        return configured
    return VSCODE_EXTENSION_MODE


def is_vscode_execution_mode(state: Mapping[str, Any] | None) -> bool:
    mode = normalize_execution_mode((state or {}).get("execution_mode"))
    return mode == VSCODE_EXTENSION_MODE


def local_python_command() -> str:
    command = str(settings.LOCAL_PYTHON_COMMAND or "python").strip()
    return command or "python"


def local_python_for_state(state: Mapping[str, Any] | None) -> str:
    source = state or {}
    configured = str(source.get("local_python_command") or local_python_command()).strip() or local_python_command()
    if not settings.EXPERIMENT_VENV_ENABLED:
        return configured
    venv_path_raw = str(source.get("venv_path") or "").strip()
    project_path_raw = str(source.get("project_path") or "").strip()
    if not venv_path_raw and project_path_raw:
        venv_path = Path(project_path_raw).expanduser().resolve() / ".venv"
    elif not venv_path_raw:
        return configured
    else:
        venv_path = Path(venv_path_raw).expanduser()
        if venv_path.name != ".venv" and project_path_raw:
            venv_path = Path(project_path_raw).expanduser().resolve() / ".venv"
    candidate = venv_path / ("Scripts" if sys.platform.startswith("win") else "bin") / (
        "python.exe" if sys.platform.startswith("win") else "python"
    )
    # In VS Code local mode, command execution happens on the user's machine.
    # Once venv is marked ready, prefer the venv python path without checking backend FS.
    if is_vscode_execution_mode(source) and bool(source.get("venv_ready")):
        return str(candidate)
    if candidate.exists():
        return str(candidate)
    return configured
