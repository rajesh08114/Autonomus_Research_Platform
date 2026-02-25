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

    topic_defaults = {
        "output_format": ".py",
        "algorithm_class": "supervised",
        "requires_quantum": False,
        "quantum_framework": "pennylane",
        "dataset_source": "synthetic",
        "kaggle_dataset_id": "owner/dataset",
        "target_metric": "accuracy",
        "hardware_target": "cpu",
        "random_seed": 42,
        "max_epochs": 20,
    }
    while state.get("status") == "waiting_user" and state.get("pending_user_question"):
        pending = state.get("pending_user_question") or {}
        current = pending.get("current_question")
        if not isinstance(current, dict):
            questions = pending.get("questions") or []
            current = questions[0] if questions else None
        assert isinstance(current, dict)
        qid = current.get("id")
        topic = str(current.get("topic", "")).strip().lower()
        answer = topic_defaults.get(topic)
        if answer is None:
            default = current.get("default")
            if default is not None:
                answer = default
            elif current.get("type") == "choice":
                options = current.get("options") or []
                answer = options[0] if options else "auto"
            elif current.get("type") == "boolean":
                answer = False
            elif current.get("type") == "number":
                answer = 1
            else:
                answer = "auto"
        state = await submit_answers(experiment_id, {qid: answer})

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
