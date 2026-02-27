from __future__ import annotations

import pytest


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
