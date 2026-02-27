from __future__ import annotations

import json
import re
from typing import Any


def strip_code_fences(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text, count=1).strip()
        text = re.sub(r"\s*```$", "", text, count=1).strip()
    return text


def _extract_braced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def parse_json_object(raw: str) -> dict[str, Any]:
    text = strip_code_fences(raw)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    snippet = _extract_braced_object(text)
    if not snippet:
        return {}
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}

