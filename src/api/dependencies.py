from __future__ import annotations

from fastapi import Request


def get_request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", "")
    return rid or "req_local"
