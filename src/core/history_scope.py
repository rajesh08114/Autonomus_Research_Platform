from __future__ import annotations

import re


def normalize_user_id(user_id: str | None) -> str:
    raw = (user_id or "").strip().lower()
    if not raw:
        return "anonymous"
    sanitized = re.sub(r"[^a-z0-9_.-]+", "_", raw).strip("_")
    if not sanitized:
        return "anonymous"
    return sanitized[:64]


def build_collection_key(user_id: str | None, test_mode: bool) -> str:
    if test_mode:
        return "test:unified"
    return f"user:{normalize_user_id(user_id)}"
