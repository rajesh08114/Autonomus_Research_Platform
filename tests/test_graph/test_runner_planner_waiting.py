from __future__ import annotations

import pytest

from src.db.repository import ExperimentRepository
from src.graph import runner as runner_module
from src.graph.runner import run_until_blocked
from src.state.research_state import ExperimentStatus, new_research_state


@pytest.mark.asyncio
async def test_run_until_blocked_keeps_planner_phase_when_waiting(tmp_path, monkeypatch):
    async def _noop(*args, **kwargs):
        _ = (args, kwargs)
        return None

    async def _llm_totals(*args, **kwargs):
        _ = (args, kwargs)
        return {"total_tokens": 0, "total_cost": 0.0}

    monkeypatch.setattr(ExperimentRepository, "add_log", _noop)
    monkeypatch.setattr(ExperimentRepository, "update", _noop)
    monkeypatch.setattr(ExperimentRepository, "get_experiment_llm_totals", _llm_totals)

    state = new_research_state("exp_runner_planner_wait", str(tmp_path), "build classifier", {"execution_mode": "vscode_extension"})
    state["status"] = ExperimentStatus.RUNNING.value
    state["phase"] = "planner"

    updated = await run_until_blocked(state)
    assert updated["status"] == ExperimentStatus.WAITING.value
    assert updated["phase"] == "planner"
    pending = updated.get("pending_user_confirm") or {}
    assert pending.get("phase") == "planner"
    assert pending.get("next_phase") == "env_manager"


@pytest.mark.asyncio
async def test_run_until_blocked_auto_answers_when_default_allow_research_enabled(tmp_path, monkeypatch):
    async def _noop(*args, **kwargs):
        _ = (args, kwargs)
        return None

    async def _llm_totals(*args, **kwargs):
        _ = (args, kwargs)
        return {"total_tokens": 0, "total_cost": 0.0}

    async def _no_more_questions(*args, **kwargs):
        _ = (args, kwargs)
        return None

    async def _planner_wait(state):
        state["status"] = ExperimentStatus.WAITING.value
        state["pending_user_confirm"] = {
            "action_id": "act_test",
            "action": "apply_file_operations",
            "phase": "planner",
            "next_phase": "env_manager",
        }
        return state

    class _ValidationResult:
        ok = True
        warnings: list[str] = []
        errors: list[str] = []

    monkeypatch.setattr(ExperimentRepository, "add_log", _noop)
    monkeypatch.setattr(ExperimentRepository, "update", _noop)
    monkeypatch.setattr(ExperimentRepository, "get_experiment_llm_totals", _llm_totals)
    monkeypatch.setattr(runner_module, "regenerate_pending_question_state", _no_more_questions)
    monkeypatch.setattr(runner_module, "planner_agent_node", _planner_wait)
    monkeypatch.setattr(runner_module, "validate_phase_output", lambda *args, **kwargs: _ValidationResult())

    state = new_research_state(
        "exp_runner_auto_clarifier",
        str(tmp_path),
        "build classifier",
        {"execution_mode": "vscode_extension", "default_allow_research": True},
    )
    state["status"] = ExperimentStatus.WAITING.value
    state["phase"] = "clarifier"
    state["pending_user_question"] = {
        "mode": "sequential_dynamic",
        "current_question": {
            "id": "Q_OUTPUT_FORMAT",
            "topic": "output_format",
            "text": "Choose output format",
            "type": "choice",
            "options": ["hybrid", ".py"],
            "required": True,
        },
        "questions": [
            {
                "id": "Q_OUTPUT_FORMAT",
                "topic": "output_format",
                "text": "Choose output format",
                "type": "choice",
                "options": ["hybrid", ".py"],
                "required": True,
            }
        ],
        "question_plan": [],
        "asked_question_ids": [],
        "answered": [],
        "answered_count": 0,
        "total_questions_planned": 1,
    }

    updated = await run_until_blocked(state)
    assert updated.get("pending_user_question") is None
    assert (updated.get("clarifications") or {}).get("output_format") == ".py"
