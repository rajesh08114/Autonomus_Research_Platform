from __future__ import annotations

import pytest

from src.core import prompt_domain


@pytest.mark.asyncio
async def test_validate_supported_prompt_accepts_ai_prompt(monkeypatch: pytest.MonkeyPatch):
    async def _fake_invoke(*args, **kwargs):
        _ = (args, kwargs)
        return '{"domain":"ai","reason":"AI task","confidence":0.9}'

    monkeypatch.setattr(prompt_domain, "invoke_master_llm", _fake_invoke)
    ok, resolved, reason = await prompt_domain.validate_supported_prompt(
        "Build an ML classifier with dataset preprocessing and f1 evaluation"
    )
    assert ok is True
    assert resolved == "ai"
    assert reason == "AI task"


@pytest.mark.asyncio
async def test_validate_supported_prompt_accepts_quantum_prompt(monkeypatch: pytest.MonkeyPatch):
    async def _fake_invoke(*args, **kwargs):
        _ = (args, kwargs)
        return '{"domain":"quantum","reason":"Quantum task","confidence":0.95}'

    monkeypatch.setattr(prompt_domain, "invoke_master_llm", _fake_invoke)
    ok, resolved, reason = await prompt_domain.validate_supported_prompt("Design a quantum circuit with 4 qubits using qiskit")
    assert ok is True
    assert resolved == "quantum"
    assert reason == "Quantum task"


@pytest.mark.asyncio
async def test_validate_supported_prompt_rejects_non_supported_prompt(monkeypatch: pytest.MonkeyPatch):
    async def _fake_invoke(*args, **kwargs):
        _ = (args, kwargs)
        return '{"domain":"unsupported","reason":"Not an AI or Quantum task","confidence":0.93}'

    monkeypatch.setattr(prompt_domain, "invoke_master_llm", _fake_invoke)
    ok, resolved, reason = await prompt_domain.validate_supported_prompt("Write a wedding invitation and travel itinerary")
    assert ok is False
    assert resolved == "ai"
    assert "Not an AI or Quantum task" in reason


@pytest.mark.asyncio
async def test_resolve_research_type_prefers_quantum_when_prompt_is_quantum(monkeypatch: pytest.MonkeyPatch):
    async def _fake_invoke(*args, **kwargs):
        _ = (args, kwargs)
        return '{"domain":"quantum","reason":"Quantum task","confidence":0.94}'

    monkeypatch.setattr(prompt_domain, "invoke_master_llm", _fake_invoke)
    resolved = await prompt_domain.resolve_research_type_from_prompt(
        "Implement a quantum ml model with a qiskit backend",
        requested="ai",
    )
    assert resolved == "quantum"
