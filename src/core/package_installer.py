from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class InstallResult:
    returncode: int
    stdout: str
    stderr: str


def parse_package_spec(spec: str) -> tuple[str, str]:
    if "==" in spec:
        package, version = spec.split("==", 1)
        return package.strip(), version.strip()
    return spec.strip(), ""


def dry_run_install(package: str, version: str, flags: list[str] | None = None) -> InstallResult:
    cmd = [sys.executable, "-m", "pip", "install", f"{package}=={version}" if version else package, "--dry-run", "--quiet"]
    if flags:
        cmd.extend(flags)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return InstallResult(proc.returncode, proc.stdout, proc.stderr)


def install_package(package: str, version: str, flags: list[str] | None = None) -> InstallResult:
    cmd = [sys.executable, "-m", "pip", "install", f"{package}=={version}" if version else package]
    if flags:
        cmd.extend(flags)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return InstallResult(proc.returncode, proc.stdout, proc.stderr)

