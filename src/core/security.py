from __future__ import annotations

import os
import re
from pathlib import Path


def validate_path(path: str, project_path: str) -> bool:
    resolved = os.path.realpath(path)
    return resolved.startswith(os.path.realpath(project_path))


def ensure_project_path(path: str, project_path: str) -> Path:
    resolved = Path(path).resolve()
    project = Path(project_path).resolve()
    if project == resolved or project in resolved.parents:
        return resolved
    raise PermissionError(f"Path traversal blocked: {resolved}")


def sanitize_subprocess_args(args: list[str]) -> list[str]:
    safe: list[str] = []
    for arg in args:
        if not re.match(r"^[a-zA-Z0-9/_\-.=@:]+$", arg):
            raise ValueError(f"Unsafe subprocess argument: {arg}")
        safe.append(arg)
    return safe

