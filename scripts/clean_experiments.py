from __future__ import annotations

from pathlib import Path
import shutil

from src.config.settings import settings


def main() -> None:
    root = settings.project_root_path
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    db = settings.state_db_path
    if db.exists():
        db.unlink()
    print("Workspace cleaned")


if __name__ == "__main__":
    main()

