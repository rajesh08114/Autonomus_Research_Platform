from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
import uuid

from src.config.settings import settings
from src.core.execution_mode import is_vscode_execution_mode, local_python_for_state
from src.core.logger import get_logger
from src.core.security import ensure_project_path, sanitize_subprocess_args
from src.db.repository import ExperimentRepository
from src.state.research_state import ExperimentStatus, ResearchState

logger = get_logger(__name__)


def _dedupe_local_file_plan(state: ResearchState) -> list[dict[str, str]]:
    latest_by_path: dict[str, dict[str, str]] = {}
    for item in state.get("local_file_plan", []):
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        latest_by_path[path] = {
            "path": path,
            "content": str(item.get("content", "")),
            "phase": str(item.get("phase", "unknown")),
        }
    return list(latest_by_path.values())


def _should_inject_failure(point: str) -> bool:
    if not settings.FAILURE_INJECTION_ENABLED:
        return False
    configured = {p.strip().lower() for p in settings.FAILURE_INJECTION_POINTS.split(",") if p.strip()}
    if configured and point.lower() not in configured:
        return False
    return random.random() < float(max(0.0, min(1.0, settings.FAILURE_INJECTION_RATE)))


def _injected_failure_stderr(experiment_id: str) -> str:
    options = [
        "SyntaxError: invalid syntax in generated script",
        "ModuleNotFoundError: No module named 'simulated_missing_dep'",
        "RuntimeError: Simulated quantum backend timeout",
        "MemoryError: Simulated memory cap breach",
        "ValueError: Simulated corrupted dataset format",
    ]
    idx = int(time.time()) % len(options)
    return f"[failure-injection:{experiment_id}] {options[idx]}"


def _resolve_experiment_python(state: ResearchState) -> str:
    if not settings.EXPERIMENT_VENV_ENABLED:
        return sys.executable
    venv_path = Path(state.get("venv_path", ""))
    if not venv_path:
        return sys.executable
    candidate = venv_path / ("Scripts" if sys.platform.startswith("win") else "bin") / ("python.exe" if sys.platform.startswith("win") else "python")
    return str(candidate) if candidate.exists() else sys.executable


def _contains_dangerous_patterns(script_path: str) -> str | None:
    blocked_markers = [
        "import requests",
        "import socket",
        "subprocess.Popen(",
        "os.system(",
        "eval(",
        "exec(",
    ]
    try:
        text = Path(script_path).read_text(encoding="utf-8")
    except Exception:
        return None
    for marker in blocked_markers:
        if marker in text:
            return marker
    return None

def classify_error(stderr: str) -> str:
    markers = [
        "ModuleNotFoundError",
        "ImportError",
        "NameError",
        "TypeError",
        "KeyError",
        "SyntaxError",
        "AttributeError",
        "ValueError",
        "RuntimeError",
        "FileNotFoundError",
        "MemoryError",
    ]
    for marker in markers:
        if marker in stderr:
            return marker
    if "CUDA" in stderr:
        return "gpu_unavailable"
    if "shape" in stderr.lower():
        return "shape_mismatch"
    if "kaggle" in stderr.lower():
        return "kaggle_auth"
    if "TimeoutExpired" in stderr:
        return "timeout"
    return "unknown"


def extract_line_number(stderr: str) -> int:
    for line in stderr.splitlines():
        if "line " in line:
            try:
                return int(line.split("line ")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                continue
    return 0


async def subprocess_runner_node(state: ResearchState) -> ResearchState:
    if not state["execution_order"]:
        return state

    if state["execution_logs"] and int(state["execution_logs"][-1]["returncode"]) != 0:
        # Preserve failure handling path in runner; do not queue the next script.
        return state

    idx = len(state["execution_logs"])
    if idx >= len(state["execution_order"]):
        return state

    script = state["execution_order"][idx]
    state["current_script"] = script
    logger.info("subprocess.run.start", experiment_id=state["experiment_id"], script=script, idx=idx)
    ensure_project_path(script, state["project_path"])
    project_path = Path(state["project_path"]).resolve()

    if is_vscode_execution_mode(state):
        local_python = local_python_for_state(state)
        file_plan = _dedupe_local_file_plan(state)
        materialized = set(state.get("local_materialized_files", []))
        materialize = [item for item in file_plan if item["path"] not in materialized]
        command = [local_python, script]
        existing = state.get("pending_user_confirm") or {}
        existing_action = str(existing.get("action", ""))
        existing_script = str(existing.get("script_path", ""))
        existing_phase = str(existing.get("phase", ""))
        existing_commands_raw = existing.get("commands", [])
        if isinstance(existing_commands_raw, list):
            existing_commands = [str(x) for x in existing_commands_raw]
        else:
            existing_commands = [str(existing_commands_raw)] if str(existing_commands_raw).strip() else []
        if (
            existing_action == "run_local_commands"
            and existing_phase == "subprocess_runner"
            and existing_script == str(script)
            and existing_commands == command
        ):
            state["status"] = ExperimentStatus.WAITING.value
            return state
        state["pending_user_confirm"] = {
            "action_id": f"act_{uuid.uuid4().hex[:8]}",
            "action": "run_local_commands",
            "phase": "subprocess_runner",
            "script_path": script,
            "cwd": str(project_path),
            "commands": command,
            "timeout_seconds": int(settings.SUBPROCESS_TIMEOUT),
            "file_operations": materialize,
            "created_files": [item["path"] for item in materialize],
            "reason": "Create project files locally and run the script in the user Python environment",
            "next_phase": "subprocess_runner",
        }
        state["status"] = ExperimentStatus.WAITING.value
        state["confirmations_requested"] = int(state.get("confirmations_requested", 0)) + 1
        await ExperimentRepository.add_log(
            state["experiment_id"],
            "subprocess_runner",
            "info",
            "Local execution requested via VS Code extension",
            {
                "script_path": script,
                "commands": command,
                "cwd": str(project_path),
                "planned_file_count": len(materialize),
                "execution_mode": "vscode_extension",
            },
        )
        return state

    log_path = project_path / "logs" / f"{Path(script).stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONPATH": str(project_path),
        "RANDOM_SEED": str(state["random_seed"]),
        "EXPERIMENT_ID": state["experiment_id"],
    }
    args = sanitize_subprocess_args([])
    executor = _resolve_experiment_python(state)
    command = [executor, script, *args]
    start = time.time()
    host = socket.gethostname()

    blocked_marker = _contains_dangerous_patterns(script)
    if blocked_marker:
        log = {
            "script_path": script,
            "command": command,
            "cwd": str(project_path),
            "returncode": -98,
            "stdout": "",
            "stderr": f"Blocked dangerous pattern in script: {blocked_marker}",
            "duration_sec": 0.0,
            "timestamp": time.time(),
            "executor": executor,
            "host": host,
        }
        with log_path.open("w", encoding="utf-8") as handle:
            json.dump(log, handle, indent=2)
        state["execution_logs"].append(log)
        state["errors"].append(
            {
                "category": "SecurityBlockedImport",
                "message": log["stderr"],
                "file_path": script,
                "line_number": 0,
                "traceback": log["stderr"],
                "timestamp": time.time(),
            }
        )
        await ExperimentRepository.add_log(
            state["experiment_id"],
            "subprocess_runner",
            "error",
            "Subprocess blocked by security policy",
            {"script_path": script, "blocked_marker": blocked_marker},
        )
        return state
    await ExperimentRepository.add_log(
        state["experiment_id"],
        "subprocess_runner",
        "info",
        "Subprocess execution started",
        {
            "script_path": script,
            "command": command,
            "cwd": str(project_path),
            "executor": executor,
            "host": host,
        },
    )

    if _should_inject_failure("subprocess_runner"):
        log = {
            "script_path": script,
            "command": command,
            "cwd": str(project_path),
            "returncode": -99,
            "stdout": "",
            "stderr": _injected_failure_stderr(state["experiment_id"]),
            "duration_sec": 0.01,
            "timestamp": time.time(),
            "executor": sys.executable,
            "host": host,
        }
    else:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=settings.SUBPROCESS_TIMEOUT,
                cwd=str(project_path),
                env=env,
            )
            elapsed = time.time() - start
            log = {
                "script_path": script,
                "command": command,
                "cwd": str(project_path),
                "returncode": result.returncode,
                "stdout": result.stdout[-settings.STDOUT_CAP_CHARS :],
                "stderr": result.stderr[-settings.STDERR_CAP_CHARS :],
                "duration_sec": elapsed,
                "timestamp": time.time(),
                "executor": executor,
                "host": host,
            }
        except subprocess.TimeoutExpired:
            log = {
                "script_path": script,
                "command": command,
                "cwd": str(project_path),
                "returncode": -1,
                "stdout": "",
                "stderr": "TimeoutExpired",
                "duration_sec": float(settings.SUBPROCESS_TIMEOUT),
                "timestamp": time.time(),
                "executor": executor,
                "host": host,
            }

    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
    state["execution_logs"].append(log)
    logger.info(
        "subprocess.run.end",
        experiment_id=state["experiment_id"],
        script=script,
        returncode=log["returncode"],
        duration_sec=log["duration_sec"],
    )
    await ExperimentRepository.add_log(
        state["experiment_id"],
        "subprocess_runner",
        "info" if log["returncode"] == 0 else "error",
        "Subprocess execution finished",
        {
            "script_path": script,
            "command": log["command"],
            "cwd": log["cwd"],
            "returncode": log["returncode"],
            "duration_sec": log["duration_sec"],
            "stdout_tail": log["stdout"][-500:],
            "stderr_tail": log["stderr"][-500:],
            "executor": log["executor"],
            "host": log["host"],
        },
    )

    if log["returncode"] != 0:
        state["errors"].append(
            {
                "category": classify_error(log["stderr"]),
                "message": log["stderr"][:1000],
                "file_path": script,
                "line_number": extract_line_number(log["stderr"]),
                "traceback": log["stderr"],
                "timestamp": time.time(),
            }
        )
        logger.warning("subprocess.run.error", experiment_id=state["experiment_id"], category=classify_error(log["stderr"]))
    return state
