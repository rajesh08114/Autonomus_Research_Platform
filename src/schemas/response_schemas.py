from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def response_envelope(success: bool, data: Any = None, error: dict | None = None, request_id: str | None = None) -> dict:
    return {
        "success": success,
        "data": data,
        "error": error,
        "request_id": request_id or f"req_{uuid.uuid4().hex[:12]}",
        "timestamp": utc_now_iso(),
        "version": "2.0.0",
    }


def error_payload(code: str, message: str, details: dict | None = None) -> dict:
    return {"code": code, "message": message, "details": details or {}}

