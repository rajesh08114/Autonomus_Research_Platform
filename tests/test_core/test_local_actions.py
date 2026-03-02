from __future__ import annotations

from src.core.local_actions import queue_local_file_action
from src.state.research_state import new_research_state


def test_queue_local_file_action_does_not_skip_writes_for_materialized_paths(tmp_path):
    state = new_research_state(
        "exp_local_actions",
        str(tmp_path),
        "Generate code",
        {"execution_mode": "vscode_extension"},
    )
    target_file = str((tmp_path / "src" / "model.py").resolve())
    target_dir = str((tmp_path / "src").resolve())
    state["local_materialized_files"] = [target_file, target_dir]

    queued = queue_local_file_action(
        state=state,
        phase="code_generator",
        file_operations=[
            {"path": target_dir, "mode": "mkdir", "phase": "code_generator"},
            {"path": target_file, "mode": "write", "content": "print('new code')\n", "phase": "code_generator"},
        ],
        next_phase="job_scheduler",
        reason="Write generated code",
        cwd=state["project_path"],
    )

    assert queued is True
    pending = state.get("pending_user_confirm") or {}
    ops = pending.get("file_operations") or []
    assert len(ops) == 1
    assert str(ops[0].get("path")) == target_file
    assert str(ops[0].get("mode")) == "write"
