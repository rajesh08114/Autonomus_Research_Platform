from __future__ import annotations

import re
import time
from typing import Any

import httpx

from src.config.settings import settings
from src.db.repository import ExperimentRepository


def _estimate_tokens(text: str) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, int(len(value) / 4))


def _compact(value: object, max_len: int = 180) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _keyword_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(token) >= 3}


def state_to_summary(state: dict[str, Any]) -> dict[str, Any]:
    evaluation = ((state.get("metrics") or {}).get("evaluation") or {}) if isinstance(state.get("metrics"), dict) else {}
    target_metric = str(state.get("target_metric") or "accuracy")
    return {
        "experiment_id": state.get("experiment_id"),
        "prompt": _compact(state.get("user_prompt"), max_len=220),
        "status": state.get("status"),
        "phase": state.get("phase"),
        "requires_quantum": bool(state.get("requires_quantum")),
        "framework": state.get("framework"),
        "quantum_framework": state.get("quantum_framework"),
        "target_metric": target_metric,
        "primary_metric_value": evaluation.get(target_metric),
        "plots_generated": list(state.get("plots_generated") or []),
        "created_at": state.get("timestamp_start"),
    }


def select_relevant_history(question: str, history: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    terms = _keyword_set(question)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, item in enumerate(history):
        haystack = " ".join(
            [
                str(item.get("prompt") or ""),
                str(item.get("framework") or ""),
                str(item.get("quantum_framework") or ""),
                str(item.get("target_metric") or ""),
                "quantum" if item.get("requires_quantum") else "classical",
            ]
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        scored.append((score, -idx, item))
    scored.sort(reverse=True, key=lambda row: (row[0], row[1]))
    selected = [row[2] for row in scored[:limit]]
    if selected:
        return selected
    return history[:limit]


def _history_to_text(selected: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in selected:
        lines.append(
            f"- {item.get('experiment_id')}: status={item.get('status')} phase={item.get('phase')} "
            f"framework={item.get('framework')} quantum={item.get('requires_quantum')} "
            f"target={item.get('target_metric')} value={item.get('primary_metric_value')} "
            f"prompt={item.get('prompt')}"
        )
    return "\n".join(lines)


async def _invoke_huggingface_chat(
    question: str,
    selected: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
) -> dict[str, Any]:
    url = f"{settings.huggingface_inference_url}/chat/completions"
    model = settings.huggingface_model_id
    history_text = _history_to_text(selected)
    recent_messages = chat_history[-6:]
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a production research copilot. Answer based on retrieved history. "
                "If data is missing, ask concise clarifying questions."
            ),
        }
    ]
    for item in recent_messages:
        role = str(item.get("role") or "user")
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": str(item.get("message") or "")[:800]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved research history:\n{history_text}\n\n"
                "Respond with concrete action-ready guidance."
            ),
        }
    )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": min(max(float(settings.MASTER_LLM_TEMPERATURE), 0.0), 0.4),
        "max_tokens": max(64, int(settings.MASTER_LLM_MAX_TOKENS)),
    }

    started = time.time()
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.huggingface_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
    latency_ms = (time.time() - started) * 1000.0

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("Chat completion response missing choices")
    content = str((choices[0].get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("Chat completion response missing content")

    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    if prompt_tokens == 0:
        prompt_tokens = _estimate_tokens(messages[-1]["content"])
    if completion_tokens == 0:
        completion_tokens = _estimate_tokens(content)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    await ExperimentRepository.add_llm_usage(
        provider="huggingface",
        model=model,
        phase="chat_assistant",
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=0.0,
        success=True,
    )
    return {
        "answer": content,
        "follow_up_questions": [],
        "generation": {
            "provider": "huggingface",
            "model": model,
            "strategy": "history_grounded_chat_completion",
            "latency_ms": round(latency_ms, 3),
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": 0.0,
        },
    }


async def generate_chat_response(
    question: str,
    selected_history: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    test_mode: bool,
) -> dict[str, Any]:
    _ = test_mode
    provider = settings.effective_master_llm_provider
    if provider == "huggingface" and settings.huggingface_api_key:
        try:
            return await _invoke_huggingface_chat(question, selected_history, chat_history)
        except Exception as exc:
            await ExperimentRepository.add_llm_usage(
                provider="huggingface",
                model=settings.huggingface_model_id,
                phase="chat_assistant",
                latency_ms=0.0,
                prompt_tokens=_estimate_tokens(question),
                completion_tokens=0,
                total_tokens=_estimate_tokens(question),
                estimated_cost_usd=0.0,
                success=False,
                error_message=str(exc),
            )
            raise RuntimeError(f"Hugging Face chat unavailable: {exc}") from exc
    raise RuntimeError(
        "Chat assistant requires Hugging Face LLM. Configure HF_API_KEY (or MASTER_LLM_API_KEY)."
    )
