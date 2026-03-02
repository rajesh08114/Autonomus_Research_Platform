from __future__ import annotations

import pytest

from src.config.settings import settings
from src.db.repository import ExperimentRepository
from src.graph import runner


@pytest.mark.asyncio
async def test_start_research(client):
    response = await client.post(
        "/api/v1/research/start",
        json={"prompt": "Build a quantum classifier for iris dataset", "config_overrides": {}},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert "experiment_id" in body["data"]


@pytest.mark.asyncio
async def test_start_research_rejects_unsupported_prompt_domain(client):
    response = await client.post(
        "/api/v1/research/start",
        json={"prompt": "Plan a 10 day vacation itinerary across europe", "config_overrides": {}},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["code"] == "UNSUPPORTED_RESEARCH_DOMAIN"


@pytest.mark.asyncio
async def test_start_research_quantum_prompt_forces_quantum_type(client):
    response = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Benchmark a quantum circuit classifier using qiskit and qubits",
            "research_type": "ai",
            "config_overrides": {},
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["data"]["research_type"] == "quantum"


@pytest.mark.asyncio
async def test_start_research_quantum_type_uses_master_llm(client):
    response = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Design a quantum circuit classifier for synthetic features",
            "research_type": "quantum",
            "config_overrides": {},
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["data"]["research_type"] == "quantum"
    assert body["data"]["llm"]["provider"] == "huggingface"


@pytest.mark.asyncio
async def test_patch_research_updates_selected_fields(client):
    start = await client.post(
        "/api/v1/research/start",
        json={"prompt": "Build a robust classifier with adaptive preprocessing", "config_overrides": {}},
    )
    assert start.status_code == 201
    experiment_id = start.json()["data"]["experiment_id"]

    patch = await client.patch(
        f"/api/v1/research/{experiment_id}",
        json={
            "updates": {
                "target_metric": "f1_macro",
                "framework": "xgboost",
                "research_plan": {"problem_type": "classification"},
            },
            "merge_nested": True,
        },
    )
    assert patch.status_code == 200
    payload = patch.json()["data"]
    assert "target_metric" in payload["updated_fields"]
    assert "framework" in payload["updated_fields"]
    assert "research_plan" in payload["updated_fields"]
    assert payload["rejected_fields"] == {}

    get_state = await client.get(f"/api/v1/research/{experiment_id}")
    assert get_state.status_code == 200
    state_payload = get_state.json()["data"]
    assert state_payload["target_metric"] == "f1_macro"
    assert state_payload["framework"] == "xgboost"


@pytest.mark.asyncio
async def test_patch_research_rejects_non_patchable_fields(client):
    start = await client.post(
        "/api/v1/research/start",
        json={"prompt": "Create an experiment for patch validation behavior", "config_overrides": {}},
    )
    assert start.status_code == 201
    experiment_id = start.json()["data"]["experiment_id"]

    patch = await client.patch(
        f"/api/v1/research/{experiment_id}",
        json={"updates": {"status": "success"}, "merge_nested": True},
    )
    assert patch.status_code == 400
    body = patch.json()
    assert body["detail"]["code"] == "INVALID_UPDATE_FIELDS"


@pytest.mark.asyncio
async def test_webhook_events_emitted_for_start_and_answer(client, monkeypatch: pytest.MonkeyPatch):
    emitted: list[dict[str, str]] = []

    async def _fake_emit(**kwargs):
        emitted.append(
            {
                "event": str(kwargs.get("event", "")),
                "experiment_id": str(kwargs.get("experiment_id", "")),
            }
        )
        return True

    monkeypatch.setattr(runner, "emit_webhook_event", _fake_emit)

    start = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Build a robust classifier with webhook notifications enabled",
            "webhook_url": "https://example.com/hooks/research",
            "config_overrides": {},
        },
    )
    assert start.status_code == 201
    payload = start.json()["data"]
    assert payload["research_scope"]["webhook_enabled"] is True

    pending = payload["pending_questions"] or {}
    question = pending.get("current_question") or {}
    question_id = str(question.get("id"))
    answer_value = question.get("default")
    if answer_value is None:
        options = question.get("options") or []
        answer_value = options[0] if options else "auto"

    answer = await client.post(
        f"/api/v1/research/{payload['experiment_id']}/answer",
        json={"answers": {question_id: answer_value}},
    )
    assert answer.status_code == 200

    events = [item["event"] for item in emitted]
    assert "experiment.started" in events
    assert "clarifier.question_required" in events
    assert "clarifier.answer_received" in events


@pytest.mark.asyncio
async def test_webhook_events_emitted_for_confirm(client, monkeypatch: pytest.MonkeyPatch):
    emitted: list[dict[str, str]] = []

    async def _fake_emit(**kwargs):
        emitted.append(
            {
                "event": str(kwargs.get("event", "")),
                "experiment_id": str(kwargs.get("experiment_id", "")),
            }
        )
        return True

    monkeypatch.setattr(runner, "emit_webhook_event", _fake_emit)
    monkeypatch.setattr(settings, "WORKFLOW_BACKGROUND_ENABLED", False)

    start = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Build a classifier and test confirmation webhook flow",
            "webhook_url": "https://example.com/hooks/research",
            "config_overrides": {},
        },
    )
    assert start.status_code == 201
    experiment_id = start.json()["data"]["experiment_id"]

    state = await ExperimentRepository.get(experiment_id)
    assert state is not None
    state["phase"] = "env_manager"
    state["status"] = "waiting_user"
    state["confirmations_requested"] = int(state.get("confirmations_requested", 0)) + 1
    state["pending_user_confirm"] = {
        "action_id": "act_test_confirm",
        "action": "install_package",
        "package": "numpy",
        "version": "2.0.0",
        "phase": "env_manager",
    }
    await ExperimentRepository.update(experiment_id, state)

    confirm = await client.post(
        f"/api/v1/research/{experiment_id}/confirm",
        json={"action_id": "act_test_confirm", "decision": "deny", "reason": "skip in test"},
    )
    assert confirm.status_code == 200

    events = [item["event"] for item in emitted]
    assert "confirmation.processed" in events
    assert "experiment.status_changed" in events
