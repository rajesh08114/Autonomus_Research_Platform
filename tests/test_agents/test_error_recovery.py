from __future__ import annotations

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

