from __future__ import annotations

import uuid
from typing import Any

from src.state.research_state import ExperimentStatus, ResearchState


def queue_local_file_action(
    state: ResearchState,
    phase: str,
    file_operations: list[dict[str, str]],
    next_phase: str,
    reason: str,
    commands: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int | None = None,
) -> bool:
    materialized = set(state.get("local_materialized_files", []))
    pending_files = [item for item in file_operations if str(item.get("path", "")) not in materialized]
    commands = [str(item) for item in (commands or []) if str(item).strip()]
    if not pending_files and not commands:
        return False

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
