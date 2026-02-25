from __future__ import annotations

import json
from typing import Any

import pytest
import httpx

from src.api.app import create_app
from src.agents import base_agent, clarifier_agent, code_gen_agent, planner_agent
from src.config.settings import settings
from src.core import chat_assistant


async def _fake_invoke_master_llm(
    system_prompt: str,
    user_prompt: str = "",
    experiment_id: str | None = None,
    phase: str | None = None,
) -> str:
    _ = (experiment_id, phase)
    context = f"{system_prompt}\n{user_prompt}".lower()

    if "parameters.questions" in context or "clarification agent" in context:
        payload = {
            "parameters": {
                "questions": [
                    {
                        "topic": "output_format",
                        "text": "Do you want .py scripts or .ipynb notebooks?",
                        "type": "choice",
                        "options": [".py", ".ipynb"],
                        "default": ".py",
                    },
                    {
                        "topic": "algorithm_class",
                        "text": "Which learning style should be prioritized?",
                        "type": "choice",
                        "options": ["supervised", "unsupervised", "reinforcement"],
                        "default": "supervised",
                    },
                ]
            }
        }
        return json.dumps(payload)

    if "methodology_additions" in context and "risk_checks" in context:
        payload = {
            "methodology_additions": ["calibrate decision threshold on validation data"],
            "risk_checks": ["verify train/test split reproducibility with fixed random seed"],
            "package_additions": ["seaborn==0.13.2"],
            "algorithm_override": "",
        }
        return json.dumps(payload)

    if "implementation_focus" in context and "failure_prevention" in context:
        payload = {
            "implementation_focus": "Build modular training and evaluation functions with explicit typed inputs.",
            "evaluation_focus": "Track primary metric per epoch and persist confusion diagnostics.",
            "failure_prevention": "Validate dataset schema and file paths before starting training.",
        }
        return json.dumps(payload)

    return json.dumps(
        {
            "action": "continue",
            "reasoning": "Unit-test LLM stub response",
            "parameters": {},
            "next_step": "planner",
            "confidence": 0.9,
        }
    )


async def _fake_chat_hf(
    question: str,
    selected: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
) -> dict[str, Any]:
    _ = (chat_history,)
    references = [str(item.get("experiment_id")) for item in selected if item.get("experiment_id")]
    answer = f"History-grounded response for: {question}. References: {', '.join(references[:3]) or 'none'}."
    prompt_tokens = max(1, int(len(question) / 4))
    completion_tokens = max(1, int(len(answer) / 4))
    return {
        "answer": answer,
        "follow_up_questions": [],
        "generation": {
            "provider": "huggingface",
            "model": settings.huggingface_model_id,
            "strategy": "history_grounded_chat_completion",
            "latency_ms": 1.0,
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": 0.0,
        },
    }


@pytest.fixture(autouse=True)
def _stub_llm_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MASTER_LLM_PROVIDER", "huggingface")
    monkeypatch.setattr(settings, "HF_API_KEY", "test_hf_key")
    monkeypatch.setattr(settings, "ALLOW_RULE_BASED_FALLBACK", False)

    monkeypatch.setattr(base_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(clarifier_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(planner_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(code_gen_agent, "invoke_master_llm", _fake_invoke_master_llm)
    monkeypatch.setattr(chat_assistant, "_invoke_huggingface_chat", _fake_chat_hf)


@pytest.fixture()
async def client() -> httpx.AsyncClient:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
