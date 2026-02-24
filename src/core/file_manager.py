from __future__ import annotations

from pathlib import Path
import json
import shutil

from src.core.security import ensure_project_path


def ensure_dirs(project_path: str, directories: list[str]) -> None:
    for rel in directories:
        (Path(project_path) / rel).mkdir(parents=True, exist_ok=True)


def write_text_file(project_path: str, file_path: str, content: str) -> None:
    target = ensure_project_path(file_path, project_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def read_text_file(project_path: str, file_path: str) -> str:
    target = ensure_project_path(file_path, project_path)
    return target.read_text(encoding="utf-8")


def replace_in_file(project_path: str, file_path: str, find: str, replace: str, backup: bool = True) -> bool:
    target = ensure_project_path(file_path, project_path)
    original = target.read_text(encoding="utf-8")
    if find not in original:
        return False
    if backup:
        shutil.copyfile(target, target.with_suffix(target.suffix + ".bak"))
    updated = original.replace(find, replace, 1)
    target.write_text(updated, encoding="utf-8")
    return True


def append_json_line(project_path: str, file_path: str, payload: dict) -> None:
    target = ensure_project_path(file_path, project_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def list_project_files(project_path: str) -> list[Path]:
    base = Path(project_path).resolve()
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file()]

