from __future__ import annotations

import pytest

from src.db.database import init_db
from src.graph.runner import get_experiment_or_404, start_experiment, submit_answers


@pytest.mark.asyncio
async def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "state.db"))
    await init_db()
    experiment_id = await start_experiment("Build a simple classifier on synthetic data", {})
    state = await get_experiment_or_404(experiment_id)
    assert state["pending_user_question"] is not None

    answer_defaults = {
        "Q1": ".py",
        "Q2": "supervised",
        "Q3": False,
        "Q4": "pennylane",
        "Q5": "synthetic",
        "Q6": "owner/dataset",
        "Q7": "accuracy",
        "Q8": "cpu",
        "Q10": 42,
        "Q11": 20,
    }
    while state.get("status") == "waiting_user" and state.get("pending_user_question"):
        pending = state.get("pending_user_question") or {}
        current = pending.get("current_question")
        if not isinstance(current, dict):
            questions = pending.get("questions") or []
            current = questions[0] if questions else None
        assert isinstance(current, dict)
        qid = current.get("id")
        assert qid in answer_defaults
        state = await submit_answers(experiment_id, {qid: answer_defaults[qid]})

    assert state["phase"] in {
        "planner",
        "env_manager",
        "finished",
        "dataset_manager",
        "code_generator",
        "job_scheduler",
        "subprocess_runner",
        "results_evaluator",
        "doc_generator",
    }
