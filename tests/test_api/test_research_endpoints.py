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
async def test_start_research_quantum_type_uses_quantum_llm(client):
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
    assert body["data"]["llm"]["provider"] == "quantum_llm"
