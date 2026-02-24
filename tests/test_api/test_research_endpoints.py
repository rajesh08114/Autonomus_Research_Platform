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
