from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_scope_user_vs_test_mode(client):
    start_alice = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Build an AI classifier with preprocessing pipeline",
            "user_id": "alice",
            "test_mode": False,
            "config_overrides": {},
        },
    )
    assert start_alice.status_code == 201
    exp_alice = start_alice.json()["data"]["experiment_id"]

    start_bob = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Create a separate user-specific baseline research run",
            "user_id": "bob",
            "test_mode": False,
            "config_overrides": {},
        },
    )
    assert start_bob.status_code == 201
    exp_bob = start_bob.json()["data"]["experiment_id"]

    start_test = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Create a test-mode unified collection experiment",
            "user_id": "qa-user",
            "test_mode": True,
            "config_overrides": {},
        },
    )
    assert start_test.status_code == 201
    exp_test = start_test.json()["data"]["experiment_id"]

    user_chat = await client.post(
        "/api/v1/chat/research",
        json={
            "message": "Summarize my previous research and next steps",
            "user_id": "alice",
            "test_mode": False,
            "context_limit": 5,
        },
    )
    assert user_chat.status_code == 200
    user_payload = user_chat.json()["data"]
    user_refs = [item["experiment_id"] for item in user_payload["retrieval"]["references"]]
    assert exp_alice in user_refs
    assert exp_bob not in user_refs
    assert exp_test not in user_refs
    assert user_payload["scope"]["collection_key"] == "user:alice"

    test_chat = await client.post(
        "/api/v1/chat/research",
        json={
            "message": "Summarize the unified test research history",
            "user_id": "someone",
            "test_mode": True,
            "context_limit": 5,
        },
    )
    assert test_chat.status_code == 200
    test_payload = test_chat.json()["data"]
    test_refs = [item["experiment_id"] for item in test_payload["retrieval"]["references"]]
    assert exp_test in test_refs
    assert exp_alice not in test_refs
    assert test_payload["scope"]["collection_key"] == "test:unified"


@pytest.mark.asyncio
async def test_chat_history_persisted(client):
    start = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Build reusable chat history validation experiment",
            "user_id": "history-user",
            "test_mode": False,
            "config_overrides": {},
        },
    )
    assert start.status_code == 201

    first = await client.post(
        "/api/v1/chat/research",
        json={"message": "What did we do before?", "user_id": "history-user", "test_mode": False},
    )
    assert first.status_code == 200
    assert first.json()["data"]["token_usage"]["total_tokens"] > 0

    second = await client.post(
        "/api/v1/chat/research",
        json={"message": "Give me preprocessing and plot guidance", "user_id": "history-user", "test_mode": False},
    )
    assert second.status_code == 200

    history = await client.get("/api/v1/chat/history", params={"user_id": "history-user", "test_mode": False, "limit": 20})
    assert history.status_code == 200
    body = history.json()["data"]
    assert body["count"] >= 4
    roles = [item["role"] for item in body["messages"]]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 2


@pytest.mark.asyncio
async def test_chat_general_question_can_skip_history_retrieval(client):
    start = await client.post(
        "/api/v1/research/start",
        json={
            "prompt": "Build an experiment used to test general chat fallback behavior",
            "user_id": "general-user",
            "test_mode": False,
            "config_overrides": {},
        },
    )
    assert start.status_code == 201

    chat = await client.post(
        "/api/v1/chat/research",
        json={
            "message": "What is gradient descent?",
            "user_id": "general-user",
            "test_mode": False,
            "context_limit": 5,
        },
    )
    assert chat.status_code == 200
    payload = chat.json()["data"]
    assert payload["retrieval"]["history_used"] == 0
    assert payload["retrieval"]["references"] == []
