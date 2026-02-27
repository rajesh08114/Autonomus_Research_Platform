from __future__ import annotations

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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            response_body = (response.text or "").strip().replace("\n", " ")
            if len(response_body) > 500:
                response_body = f"{response_body[:500]}..."
            raise RuntimeError(
                f"Hugging Face API returned HTTP {response.status_code} at {url} "
                f"for model '{model}'. Response: {response_body}"
            ) from exc
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
    if provider not in {"huggingface", "hf", "hugging_face"}:
        raise RuntimeError(
            "Only HuggingFace provider is supported in LLM-only mode. "
            "Set MASTER_LLM_PROVIDER=huggingface and configure HF_API_KEY (or MASTER_LLM_API_KEY)."
        )
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
        raise RuntimeError(f"Hugging Face invocation failed: {exc}") from exc


async def assert_master_llm_ready() -> None:
    """Fail fast when master LLM is not reachable at startup."""
    if not settings.huggingface_api_key:
        raise RuntimeError("HF_API_KEY (or MASTER_LLM_API_KEY) is required for startup readiness.")
    await _invoke_huggingface_chat(
        system_prompt="You are a health-check assistant. Return compact JSON.",
        user_prompt='Return exactly {"status":"ok"}',
        experiment_id=None,
        phase="startup_readiness",
    )
