from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import httpx

from src.core.logger import get_logger

logger = get_logger(__name__)

_WEBHOOK_TIMEOUT_SEC = 5.0
_WEBHOOK_MAX_RETRIES = 2
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


async def emit_webhook_event(
    webhook_url: str | None,
    event: str,
    experiment_id: str,
    status: str,
    phase: str,
    data: dict[str, Any] | None = None,
) -> bool:
    url = str(webhook_url or "").strip()
    if not url:
        return False

    payload = {
        "event_id": f"wh_{uuid.uuid4().hex[:16]}",
        "event": event,
        "timestamp": time.time(),
        "experiment_id": experiment_id,
        "status": str(status or ""),
        "phase": str(phase or ""),
        "data": data or {},
    }

    timeout = httpx.Timeout(_WEBHOOK_TIMEOUT_SEC, connect=_WEBHOOK_TIMEOUT_SEC)
    attempts = _WEBHOOK_MAX_RETRIES + 1
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(url, json=payload)
                if 200 <= response.status_code < 300:
                    logger.info(
                        "webhook.sent",
                        experiment_id=experiment_id,
                        event=event,
                        status_code=response.status_code,
                        attempt=attempt,
                    )
                    return True

                retryable = response.status_code in _RETRYABLE_STATUS_CODES
                logger.warning(
                    "webhook.http_error",
                    experiment_id=experiment_id,
                    event=event,
                    status_code=response.status_code,
                    retryable=retryable,
                    attempt=attempt,
                )
                if not retryable or attempt >= attempts:
                    return False
            except Exception as exc:
                logger.warning(
                    "webhook.request_failed",
                    experiment_id=experiment_id,
                    event=event,
                    error=str(exc),
                    attempt=attempt,
                )
                if attempt >= attempts:
                    return False

            await asyncio.sleep(min(1.5, 0.25 * attempt))
    return False
