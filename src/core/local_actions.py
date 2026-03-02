from __future__ import annotations

import uuid
from typing import Any

from src.state.research_state import ExperimentStatus, ResearchState


def _pending_signature(state: ResearchState) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...], str]:
    pending = state.get("pending_user_confirm") or {}
    action = str(pending.get("action", ""))
    phase = str(pending.get("phase", ""))
    next_phase = str(pending.get("next_phase", ""))
    files = tuple(sorted(str(item) for item in pending.get("created_files", []) if str(item).strip()))
    commands = tuple(str(item).strip() for item in pending.get("commands", []) if str(item).strip())
    cwd = str(pending.get("cwd", ""))
    return action, phase, next_phase, files, commands, cwd


def queue_local_file_action(
    state: ResearchState,
    phase: str,
    file_operations: list[dict[str, Any]],
    next_phase: str,
    reason: str,
    commands: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int | None = None,
) -> bool:
    materialized = set(state.get("local_materialized_files", []))
    pending_files: list[dict[str, Any]] = []
    for item in file_operations:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        mode = str(item.get("mode", "write")).strip().lower()
        # Directory creation can be skipped once materialized, but file writes
        # must still flow through so generated code/report content can overwrite placeholders.
        if mode in {"mkdir", "directory"} and path in materialized:
            continue
        pending_files.append(item)
    commands = [str(item) for item in (commands or []) if str(item).strip()]
    if not pending_files and not commands:
        return False

    target_signature = (
        "apply_file_operations",
        str(phase),
        str(next_phase),
        tuple(sorted(str(item.get("path", "")) for item in pending_files if str(item.get("path", "")).strip())),
        tuple(str(item).strip() for item in commands if str(item).strip()),
        str(cwd or state["project_path"]),
    )
    if _pending_signature(state) == target_signature:
        state["status"] = ExperimentStatus.WAITING.value
        return True

    state["pending_user_confirm"] = {
        "action_id": f"act_{uuid.uuid4().hex[:8]}",
        "action": "apply_file_operations",
        "phase": phase,
        "cwd": cwd or state["project_path"],
        "file_operations": pending_files,
        "created_files": [str(item.get("path", "")) for item in pending_files if str(item.get("path", "")).strip()],
        "commands": commands,
        "timeout_seconds": int(timeout_seconds or 0),
        "reason": reason,
        "next_phase": next_phase,
    }
    state["status"] = ExperimentStatus.WAITING.value
    state["confirmations_requested"] = int(state.get("confirmations_requested", 0)) + 1
    return True
