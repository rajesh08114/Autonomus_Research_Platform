from __future__ import annotations

import json
import time
from typing import Any

import httpx

from src.config.settings import settings
from src.core.logger import get_logger
from src.db.repository import ExperimentRepository

logger = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    # Rough heuristic for accounting when provider usage metadata is unavailable.
    return max(1, int(len(value) / 4))


def _local_structured_fallback() -> str:
    payload: dict[str, Any] = {
        "action": "ask_user",
        "reasoning": "Local structured fallback used while remote provider is unavailable.",
        "parameters": {"questions": []},
        "next_step": "planner",
        "confidence": 0.6,
    }
    return json.dumps(payload)


async def _invoke_huggingface_chat(
    system_prompt: str,
    user_prompt: str = "",
    experiment_id: str | None = None,
    phase: str | None = None,
) -> str:
    if not settings.huggingface_api_key:
        raise RuntimeError("HF_API_KEY (or MASTER_LLM_API_KEY) is required for huggingface provider")

    base = settings.huggingface_inference_url
    url = f"{base}/chat/completions"
    model = settings.huggingface_model_id

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt or "Return one valid JSON action object only."},
        ],
        "temperature": settings.MASTER_LLM_TEMPERATURE,
        "max_tokens": settings.MASTER_LLM_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    started = time.time()
    logger.info("llm.huggingface.request", model=model, url=url)
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
        raise RuntimeError("HuggingFace response does not contain choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("HuggingFace response missing assistant message content")
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    if prompt_tokens == 0:
        prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
    if completion_tokens == 0:
        completion_tokens = _estimate_tokens(content)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    logger.info(
        "llm.huggingface.response",
        model=model,
        content_len=len(content),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 2),
    )
    await ExperimentRepository.add_llm_usage(
        experiment_id=experiment_id,
        phase=phase,
        provider="huggingface",
        model=model,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=0.0,
        success=True,
    )
    return content


async def invoke_master_llm(
    system_prompt: str,
    user_prompt: str = "",
    experiment_id: str | None = None,
    phase: str | None = None,
) -> str:
    """Master LLM adapter.

    Uses configured provider or auto-selects best available provider.
    """
    provider = settings.effective_master_llm_provider
    logger.info("llm.invoke.start", provider=provider)
    if provider == "rule_based":
        if not settings.ALLOW_RULE_BASED_FALLBACK:
            raise RuntimeError("Rule-based fallback is disabled. Configure HF_API_KEY for autonomous LLM execution.")
        content = _local_structured_fallback()
        prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        completion_tokens = _estimate_tokens(content)
        total_tokens = prompt_tokens + completion_tokens
        await ExperimentRepository.add_llm_usage(
            experiment_id=experiment_id,
            phase=phase,
            provider="rule_based",
            model="local_structured_fallback",
            latency_ms=0.1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=0.0,
            success=True,
        )
        return content
    if provider in {"huggingface", "hf", "hugging_face"}:
        try:
            return await _invoke_huggingface_chat(
                system_prompt,
                user_prompt,
                experiment_id=experiment_id,
                phase=phase,
            )
        except Exception as exc:
            logger.exception("llm.huggingface.error")
            await ExperimentRepository.add_llm_usage(
                experiment_id=experiment_id,
                phase=phase,
                provider="huggingface",
                model=settings.huggingface_model_id,
                latency_ms=0.0,
                prompt_tokens=_estimate_tokens(system_prompt) + _estimate_tokens(user_prompt),
                success=False,
                error_message=str(exc),
            )
            if settings.ALLOW_RULE_BASED_FALLBACK:
                return _local_structured_fallback()
            raise RuntimeError(f"Hugging Face invocation failed: {exc}") from exc

    # Placeholder for provider-specific integration in production.
    raise RuntimeError(f"Configured master LLM provider '{provider}' is not implemented.")
