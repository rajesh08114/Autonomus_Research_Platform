from __future__ import annotations

import pytest

from src.agents.dataset_agent import dataset_agent_node
from src.agents.env_manager_agent import apply_user_confirmation
from src.core.subprocess_runner import subprocess_runner_node
from src.db.repository import ExperimentRepository
from src.state.research_state import ExperimentStatus, new_research_state


@pytest.mark.asyncio
async def test_subprocess_runner_emits_local_action_and_accepts_result(tmp_path, monkeypatch):
    async def _noop_add_log(*args, **kwargs):
        return None

    monkeypatch.setattr(ExperimentRepository, "add_log", _noop_add_log)

    state = new_research_state("exp_local_1", str(tmp_path), "run locally", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "subprocess_runner"
    script_path = str(tmp_path / "main.py")
    state["execution_order"] = [script_path]
    state["local_file_plan"] = [{"path": script_path, "content": "print('ok')\n", "phase": "code_generator"}]

    state = await subprocess_runner_node(state)
    pending = state.get("pending_user_confirm") or {}
    assert state["status"] == ExperimentStatus.WAITING.value
    assert pending.get("action") == "run_local_commands"
    assert pending.get("file_operations")

    action_id = str(pending.get("action_id"))
    state = await apply_user_confirmation(
        state,
        action_id=action_id,
        decision="confirm",
        execution_result={
            "returncode": 0,
            "stdout": "METRIC: accuracy=0.9",
            "stderr": "",
            "duration_sec": 1.2,
            "command": ["python", script_path],
            "cwd": str(tmp_path),
            "created_files": [script_path],
        },
    )
    assert state["status"] == ExperimentStatus.RUNNING.value
    assert state["pending_user_confirm"] is None
    assert len(state["execution_logs"]) == 1
    assert script_path in state["local_materialized_files"]


@pytest.mark.asyncio
async def test_dataset_phase_emits_apply_files_action_and_resumes_next_phase(tmp_path):
    state = new_research_state("exp_local_2", str(tmp_path), "prepare dataset", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "dataset_manager"

    state = await dataset_agent_node(state)
    pending = state.get("pending_user_confirm") or {}
    assert state["status"] == ExperimentStatus.WAITING.value
    assert pending.get("action") == "apply_file_operations"
    assert pending.get("next_phase") == "code_generator"
    created_files = [str(path) for path in pending.get("created_files", [])]
    assert created_files

    state = await apply_user_confirmation(
        state,
        action_id=str(pending.get("action_id")),
        decision="confirm",
        execution_result={
            "returncode": 0,
            "stdout": "DATA_VALID: true",
            "stderr": "",
            "duration_sec": 0.8,
            "command": ["python", str(tmp_path / "data" / "validate_data.py")],
            "cwd": str(tmp_path),
            "created_files": created_files,
        },
    )
    assert state["status"] == ExperimentStatus.RUNNING.value
    assert state["phase"] == "code_generator"
    assert state["pending_user_confirm"] is None
