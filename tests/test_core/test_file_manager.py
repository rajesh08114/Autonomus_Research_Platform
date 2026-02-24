from __future__ import annotations

from pathlib import Path

from src.core.file_manager import write_text_file, read_text_file


def test_file_roundtrip(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "x.txt"
    write_text_file(str(project), str(target), "hello")
    assert read_text_file(str(project), str(target)) == "hello"

