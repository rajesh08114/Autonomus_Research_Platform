from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.error_recovery_agent import error_recovery_agent_node
from src.state.research_state import ExperimentStatus, new_research_state


@pytest.mark.asyncio
async def test_error_recovery_aborts_after_limit(tmp_path):
    state = new_research_state("exp_1", str(tmp_path), "x", {})
    state["status"] = ExperimentStatus.RUNNING
    state["retry_count"] = 5
    state["errors"].append(
        {
            "category": "unknown",
            "message": "boom",
            "file_path": "main.py",
            "line_number": 1,
            "traceback": "boom",
            "timestamp": 0.0,
        }
    )
    state = await error_recovery_agent_node(state)
    assert str(state["status"]) == "aborted"


@pytest.mark.asyncio
async def test_error_recovery_local_execution_error_triggers_codegen_repair(tmp_path):
    state = new_research_state("exp_local_exec", str(tmp_path), "x", {})
    state["status"] = ExperimentStatus.RUNNING
    state["retry_count"] = 0
    state["research_plan"]["code_level"] = "intermediate"
    state["errors"].append(
        {
            "category": "LocalExecutionError",
            "message": "NameError: name 'something' is not defined",
            "file_path": str(tmp_path / "main.py"),
            "line_number": 1,
            "traceback": "Traceback ...",
            "timestamp": 0.0,
        }
    )

    updated = await error_recovery_agent_node(state)
    assert str(updated["phase"]) == "code_generator"
    assert int(updated["retry_count"]) == 1
    assert len(updated["repair_history"]) == 1
    assert "regeneration" in str(updated["repair_history"][0]["fix_description"]).lower()


@pytest.mark.asyncio
async def test_error_recovery_injects_logger_for_nameerror(tmp_path):
    src_dir = Path(tmp_path) / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_path = src_dir / "preprocessing.py"
    preprocessing_path.write_text(
        "from __future__ import annotations\n\n"
        "def load_and_preprocess_data():\n"
        "    logger.info('Loading and preprocessing data')\n"
        "    return {}\n",
        encoding="utf-8",
    )

    state = new_research_state("exp_logger_fix", str(tmp_path), "x", {})
    state["status"] = ExperimentStatus.RUNNING
    state["errors"].append(
        {
            "category": "LocalExecutionError",
            "message": "NameError: name 'logger' is not defined",
            "file_path": str(tmp_path / "main.py"),
            "line_number": 1,
            "traceback": (
                f'Traceback (most recent call last):\n'
                f'  File "{tmp_path / "main.py"}", line 8, in <module>\n'
                f'  File "{preprocessing_path}", line 6, in load_and_preprocess_data\n'
                "    logger.info('Loading and preprocessing data')\n"
                "NameError: name 'logger' is not defined\n"
            ),
            "timestamp": 0.0,
        }
    )

    updated = await error_recovery_agent_node(state)
    assert str(updated["phase"]) == "subprocess_runner"
    assert len(updated["repair_history"]) == 1
    patched = preprocessing_path.read_text(encoding="utf-8")
    assert "import logging" in patched
    assert "logger = logging.getLogger(__name__)" in patched
